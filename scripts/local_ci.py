#!/usr/bin/env python3
"""Local, fail-closed validation. Run inside the existing development container."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GIT = ['git', '-c', 'core.filemode=false', '-c', 'core.autocrlf=true']
PROFILES = ('fast', 'verify', 'release', 'legacy', 'doctor', 'sanitize', 'ubsan')


def capture(args, root=ROOT):
    return subprocess.check_output(args, cwd=root, text=True).strip()


def source_files(root=ROOT):
    raw = subprocess.check_output(
        GIT + ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    return sorted(set(p for p in raw.decode().split('\0') if p))


def file_hash(path):
    # A Windows checkout and its Linux bind mount must identify the same source.
    data = path.read_bytes()
    if b'\0' not in data:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def fingerprint(root=ROOT):
    digest = hashlib.sha256()
    for name in source_files(root):
        path = root / name
        value = file_hash(path) if path.is_file() else 'MISSING'
        digest.update((name + '\0' + value + '\n').encode())
    return digest.hexdigest()


def oracle_files(root=ROOT):
    prefixes = ('flightlib/src/', 'flightlib/include/', 'flightlib/configs/',
                'flightmare-parity-tests/')
    return [p for p in source_files(root)
            if p.startswith(prefixes) or p == 'flightlib/CMakeLists.txt']


def check_oracle(root=ROOT):
    expected = json.loads((root / 'scripts/oracle-baseline.json').read_text())['files']
    actual = {name: file_hash(root / name) if (root / name).is_file() else 'MISSING'
              for name in oracle_files(root)}
    changed = sorted(name for name in expected.keys() | actual.keys()
                     if expected.get(name) != actual.get(name))
    if changed:
        raise RuntimeError('Frozen oracle changed; review explicitly before updating '
                           'scripts/oracle-baseline.json:\n' + '\n'.join(changed))


class Runner:
    def __init__(self, directory, root=ROOT):
        self.directory = directory
        self.root = root
        self.steps = []

    def run(self, name, args, env=None, timeout=900, cwd=None):
        print(f'[{name}] {" ".join(map(str, args))}', flush=True)
        log = self.directory / f'{name}.log'
        step = {'name': name, 'command': list(map(str, args)), 'log': log.name}
        self.steps.append(step)
        start = time.monotonic()
        try:
            with log.open('w') as stream:
                process = subprocess.Popen(args, cwd=cwd or self.root, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    returncode = process.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    # Stop descendants too: a timed-out Ninja must not keep
                    # writing the build tree after the harness releases its lock.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            step['returncode'] = returncode
            if returncode:
                raise RuntimeError(f'{name} failed (exit {returncode})')
        except (OSError, subprocess.TimeoutExpired, RuntimeError, KeyboardInterrupt):
            step['status'] = 'failed'
            if log.exists():
                print(log.read_text(errors='replace')[-12000:], flush=True)
            raise
        else:
            step['status'] = 'passed'
        finally:
            step['seconds'] = round(time.monotonic() - start, 3)

    def cmake(self, name, source, build, options=(), targets=()):
        self.run(name + '-configure', ['cmake', '-S', source, '-B', build, '-G', 'Ninja',
                                      '-DCMAKE_BUILD_TYPE=Release', *options])
        command = ['cmake', '--build', build, '--parallel',
                   os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '4')]
        if targets:
            command += ['--target', *targets]
        self.run(name + '-build', command)

    def ctest(self, name, build):
        self.run(name, ['ctest', '--test-dir', build, '--output-on-failure',
                        '--no-tests=error', '--timeout', '120', '--output-junit',
                        str(self.directory / f'{name}.xml')])


def execute(profile, runner):
    arch = os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES', 'native')
    cuda_options = [f'-DCMAKE_CUDA_ARCHITECTURES={arch}']
    if profile in ('fast', 'verify'):
        runner.run('harness', [sys.executable, '-m', 'unittest', 'discover',
                               '-s', 'scripts/tests', '-v'])
        runner.cmake('contracts', 'flightlib-cuda', '/build/flightlib-cuda-contracts',
                     ['-DBUILD_TESTING=ON', '-DFLIGHTLIB_CUDA_TEST_DEVICE=ON',
                      '-DFLIGHTLIB_CUDA_ORACLE_LIBRARY=', *cuda_options])
        runner.ctest('contracts', '/build/flightlib-cuda-contracts')
    if profile == 'verify':
        # Build the binding against the SAME oracle library, not a second build.
        runner.cmake('parity', 'flightmare-parity-tests', '/build/flightmare-parity-tests',
                     targets=['flightmare_parity_tests', 'flightmare_oracle_trace', 'flightgym'])
        runner.ctest('parity', '/build/flightmare-parity-tests')
        runner.cmake('lifecycle', 'flightlib-cuda', '/build/flightlib-cuda-contracts',
                     ['-DBUILD_TESTING=ON', '-DFLIGHTLIB_CUDA_TEST_DEVICE=ON',
                      '-DFLIGHTLIB_CUDA_ORACLE_LIBRARY=/build/flightmare-parity-tests/flightlib/libflightlib.a',
                      *cuda_options], targets=['lifecycle_test', 'dynamics_test', 'environment_test'])
        runner.run('lifecycle-parity', ['ctest', '--test-dir', '/build/flightlib-cuda-contracts',
                   '-R', '^flightlib_cuda.(lifecycle|dynamics|environment)_parity$', '--no-tests=error',
                   '--output-on-failure', '--output-junit', str(runner.directory / 'lifecycle.xml')])
        # Eigen packet intrinsics can differ from scalar math even with strict
        # FP flags. Qualify the actual optimized CPU comparison, including rewards.
        runner.cmake('optimized-cpu', 'flightmare-profile/native', '/build/flightmare-profile',
                     ['-DPROFILE_CPU_BASELINE=ON', *cuda_options],
                     targets=['cpu_baseline', 'cpu_environment_parity', 'cpu_environment'])
        for name in ('cpu_baseline', 'cpu_environment_parity'):
            runner.run(name, [f'/build/flightmare-profile/{name}'],
                       env=dict(os.environ, FLIGHTMARE_PATH=str(ROOT)))
        cuda_python = '/build/flightlib-cuda-python/bin/python'
        if not Path(cuda_python).exists():
            raise RuntimeError('Run docker compose exec -T dev bash scripts/setup-cuda-python.sh first')
        runner.cmake('cuda-python', 'flightlib-cuda', '/build/flightlib-cuda-python-native',
                     ['-DBUILD_TESTING=OFF', '-DFLIGHTLIB_CUDA_BUILD_RUNTIME=ON',
                      '-DFLIGHTLIB_CUDA_BUILD_PYTHON=ON', f'-DPython_EXECUTABLE={cuda_python}', *cuda_options])
        runner.run('cuda-python-parity', [cuda_python, '-m', 'pytest', '-q',
                   'flightlib-cuda/python/tests', f'--junitxml={runner.directory / "cuda-python.xml"}'],
                   env=dict(os.environ, PYTHONPATH='/build/flightlib-cuda-python-native/python'))
        runner.run('rl-build', [cuda_python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', 'flightrl-cuda'])
        runner.run('rl-acceptance', [cuda_python, '-m', 'pytest', '-q', 'flightrl-cuda/tests',
                   f'--junitxml={runner.directory / "rl.xml"}'],
                   env=dict(os.environ, PYTHONPATH='/workspace/flightrl-cuda:/workspace/flightmare-profile:/build/flightlib-cuda-python-native/python'))
        # Install to a fresh destination so stale source/build imports cannot
        # masquerade as a complete standalone package.
        install = runner.directory / 'installed'
        runner.run('core-install', ['cmake', '--install', '/build/flightlib-cuda-python-native', '--prefix', str(install)])
        runner.run('installed-core-example', [cuda_python, str(ROOT/'flightlib-cuda/python/examples/core.py')],
                   env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('rl-install', [cuda_python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                                 '--target', str(install), str(ROOT/'flightrl-cuda')])
        checkpoint=runner.directory/'policy.pt'
        # Detect accidental resolution through the editable source install.
        consumer = (
            'import os, runpy, flightlib_cuda, flightrl_cuda; from pathlib import Path; '
            'root = Path(os.environ["PYTHONPATH"]).resolve(); '
            'assert all(Path(m.__file__).resolve().is_relative_to(root) '
            'for m in (flightlib_cuda, flightrl_cuda)); '
            'runpy.run_module("flightrl_cuda.custom_collector", run_name="__main__")'
        )
        runner.run('installed-custom-collector', [cuda_python, '-c', consumer],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('installed-course-rollout', [cuda_python, '-m', 'flightrl_cuda.course_rollout'],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('installed-swarm-rollout', [cuda_python, '-m', 'flightrl_cuda.course_rollout','--agent-collisions'],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('installed-sensor-example', [cuda_python, '-m', 'flightrl_cuda.sensor_example'],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('installed-transport-example', [cuda_python, '-m', 'flightrl_cuda.transport_example'],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        runner.run('isaac-adapter-install', [cuda_python, '-m', 'pip', 'install', '--no-deps',
                   '--no-build-isolation', '--target', str(install), './flightmare-isaac'])
        runner.run('installed-camera-contract', [cuda_python, '-m', 'pytest', '-q',
                   str(ROOT/'flightmare-isaac/tests'), f'--junitxml={runner.directory / "camera-contract.xml"}'],
                   cwd='/tmp', env=dict(os.environ, PYTHONPATH=str(install)))
        # Run from /tmp, with only the installed packages on PYTHONPATH.
        entry='import os,runpy; os.chdir("/tmp"); runpy.run_module("flightrl_cuda.train",run_name="__main__")'
        runner.run('installed-rl-train', [cuda_python,'-c',entry,'--envs','2','--slots','4','--steps','8','--updates','2',
                   '--checkpoint',str(checkpoint)], env=dict(os.environ,PYTHONPATH=str(install)))
        runner.run('installed-rl-evaluate', [cuda_python,'-c',entry,'--envs','2','--slots','4','--steps','8','--evaluate',
                   '--checkpoint',str(checkpoint)], env=dict(os.environ,PYTHONPATH=str(install)))
        runner.cmake('installed-native', 'flightlib-cuda/examples/native', str(runner.directory/'consumer'),
                     [f'-DCMAKE_PREFIX_PATH={install}'])
        runner.run('installed-native-consumer', [str(runner.directory/'consumer/core_consumer')])
        env = dict(os.environ, PYTHONPATH='/build/flightmare-parity-tests/flightlib')
        runner.run('binding', [sys.executable, '-m', 'pytest', '-q',
                               'docker/smoke/test_flightgym.py',
                               f'--junitxml={runner.directory / "binding.xml"}'], env=env)
    elif profile == 'legacy':
        runner.cmake('legacy', 'flightlib', '/build/flightlib',
                     ['-DFLIGHTLIB_USE_SYSTEM_DEPS=ON', '-DBUILD_TESTS=ON',
                      '-DBUILD_UNITY_BRIDGE_TESTS=OFF', '-DENABLE_FAST=OFF'])
        runner.ctest('legacy', '/build/flightlib')
    elif profile == 'doctor':
        runner.run('compiler', ['nvcc', '--version'])
        runner.run('gpu', ['nvidia-smi'])
        runner.cmake('smoke', 'docker/smoke', '/build/cuda-smoke', cuda_options)
        runner.ctest('smoke', '/build/cuda-smoke')
    elif profile == 'sanitize':
        # Fail fast on unsupported instrumentation before the expensive suite.
        # A normal executable pass is never a substitute for sanitizer success.
        runner.cmake('sanitizer-probe', 'flightlib-cuda', '/build/flightlib-cuda-contracts',
                     ['-DBUILD_TESTING=ON', '-DFLIGHTLIB_CUDA_TEST_DEVICE=ON', *cuda_options],
                     targets=['runtime_test'])
        runner.run('memcheck', ['compute-sanitizer', '--tool', 'memcheck',
                               '--error-exitcode', '1',
                               '/build/flightlib-cuda-contracts/runtime_test'])
        execute('verify', runner)
        targets = ('runtime_test', 'guarded_test', 'lifecycle_test', 'dynamics_test', 'environment_test')
        env = dict(os.environ, FLIGHTMARE_PATH=str(ROOT),
                   PYTHONPATH='/workspace/flightrl-cuda:/workspace/flightmare-profile:/build/flightlib-cuda-python-native/python')
        for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
            prefix = ['compute-sanitizer', '--tool', tool, '--error-exitcode', '1']
            for target in targets:
                if tool == 'memcheck' and target == 'runtime_test': continue
                runner.run(f'{tool}-{target}', prefix + [f'/build/flightlib-cuda-contracts/{target}'],
                           env=env, timeout=900)
            runner.run(f'{tool}-rl', prefix + ['/build/flightlib-cuda-python/bin/python', '-m', 'pytest', '-q',
                       'flightrl-cuda/tests/test_rl.py', '-k',
                       'gae_terminal or nondefault_stream or ppo_update_checkpoint'], env=env, timeout=900)
    elif profile == 'ubsan':
        runner.cmake('oracle', 'flightmare-parity-tests', '/build/flightmare-parity-tests',
                     targets=['flightlib'])
        runner.cmake('ubsan', 'flightlib-cuda', '/build/flightlib-cuda-ubsan', [
            '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
            '-DBUILD_TESTING=ON', '-DFLIGHTLIB_CUDA_TEST_DEVICE=ON',
            '-DFLIGHTLIB_CUDA_ORACLE_LIBRARY=/build/flightmare-parity-tests/flightlib/libflightlib.a',
            '-DCMAKE_CXX_FLAGS=-Wall -Wextra -Wpedantic -fsanitize=undefined -fno-omit-frame-pointer',
            '-DCMAKE_CUDA_FLAGS=-lineinfo -Xcompiler=-fsanitize=undefined -Xcompiler=-fno-omit-frame-pointer',
            '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined', *cuda_options])
        runner.run('ubsan-tests', ['ctest', '--test-dir', '/build/flightlib-cuda-ubsan',
                   '--output-on-failure', '--no-tests=error', '--timeout', '120',
                   '--output-junit', str(runner.directory / 'ubsan.xml')],
                   env=dict(os.environ, UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=PROFILES, nargs='?', default='fast')
    args = parser.parse_args()
    directory = ROOT / '.local-ci' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                                    + '-' + args.profile)
    directory.mkdir(parents=True)
    runner = Runner(directory)
    report = {'profile': args.profile, 'status': 'failed', 'steps': runner.steps,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        report['version'] = (ROOT / 'VERSION').read_text().strip()
        if not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', report['version']):
            raise RuntimeError('VERSION must contain a numeric major.minor.patch version')
        if f'## {report["version"]} ' not in (ROOT / 'CHANGELOG.md').read_text():
            raise RuntimeError('Current VERSION has no CHANGELOG entry')
        report['commit'] = capture(GIT + ['rev-parse', 'HEAD'])
        report['git_status'] = capture(GIT + ['status', '--short'])
        if args.profile == 'release':
            if report['git_status']:
                raise RuntimeError('Release validation requires a clean committed tree')
            if capture(GIT + ['tag', '--list', 'v' + report['version']]):
                raise RuntimeError('Release tag already exists; never move a version tag')
        report['source_fingerprint'] = fingerprint()
        check_oracle()
        runner.run('whitespace', GIT + ['diff', '--check', 'HEAD'])
        for name, command in [('toolchain', ['nvcc', '--version']),
                              ('cxx', ['c++', '--version']),
                              ('cmake', ['cmake', '--version']),
                              ('device', ['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                          '--format=csv'])]:
            report[name] = capture(command)
        report['cuda_architectures'] = os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES', 'native')
        report['container_hostname'] = capture(['hostname'])
        execute('verify' if args.profile == 'release' else args.profile, runner)
        if fingerprint() != report['source_fingerprint']:
            raise RuntimeError('Source changed during validation; rerun against a stable tree')
        if args.profile == 'release' and (
                capture(GIT + ['rev-parse', 'HEAD']) != report['commit'] or
                capture(GIT + ['status', '--short'])):
            raise RuntimeError('Release commit/worktree changed during validation')
        report['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = str(error)
        print(f'FAILED: {error}', file=sys.stderr)
    finally:
        report['seconds'] = round(time.monotonic() - start, 3)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{report["status"].upper()}: {directory / "report.json"}', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    # Shared build volume: concurrent reconfiguration can invalidate tests.
    with open('/build/flightmare-local-ci.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('Another local validation is using /build; wait for it to finish')
        sys.exit(main())
