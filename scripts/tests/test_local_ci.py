"""Test harness failure paths, not the simulator's implementation details."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('local_ci', Path(__file__).parents[1] / 'local_ci.py')
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class HarnessFailures(unittest.TestCase):
    def test_ctest_with_no_tests_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CTestTestfile.cmake').write_text('# Deliberately empty\n')
            runner = ci.Runner(root)
            with self.assertRaises(RuntimeError):
                runner.ctest('empty-suite', str(root))
            self.assertEqual(runner.steps[0]['status'], 'failed')

    def test_failed_process_is_not_a_passing_step(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(Path(directory))
            with self.assertRaises(RuntimeError):
                runner.run('failure', [sys.executable, '-c', 'raise SystemExit(7)'])
            self.assertEqual(runner.steps[0]['status'], 'failed')
            self.assertEqual(runner.steps[0]['returncode'], 7)

    def test_timeout_is_not_a_passing_step(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ci.Runner(Path(directory))
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run('timeout', [sys.executable, '-c', 'import time; time.sleep(10)'],
                           timeout=0.05)
            self.assertEqual(runner.steps[0]['status'], 'failed')

    def test_oracle_guard_detects_modified_added_and_deleted_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'flightlib/src').mkdir(parents=True)
            (root / 'scripts').mkdir()
            source = root / 'flightlib/src/example.cpp'
            source.write_bytes(b'original\r\n')
            expected = ci.file_hash(source)
            (root / 'scripts/oracle-baseline.json').write_text(json.dumps(
                {'files': {'flightlib/src/example.cpp': expected}}))
            ci.check_oracle(root)
            # Line ending conversion is not a physics change.
            source.write_bytes(b'original\n')
            ci.check_oracle(root)
            source.write_text('changed\n')
            with self.assertRaises(RuntimeError):
                ci.check_oracle(root)
            source.write_text('original\n')
            extra = root / 'flightlib/src/new.cpp'
            extra.write_text('new')
            with self.assertRaises(RuntimeError):
                ci.check_oracle(root)
            extra.unlink()
            source.unlink()
            with self.assertRaises(RuntimeError):
                ci.check_oracle(root)

    def test_fingerprint_includes_untracked_and_modified_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            before = ci.fingerprint(root)
            source = root / 'new.cpp'
            source.write_text('first')
            first = ci.fingerprint(root)
            self.assertNotEqual(before, first)
            source.write_text('second')
            self.assertNotEqual(first, ci.fingerprint(root))


if __name__ == '__main__':
    unittest.main()
