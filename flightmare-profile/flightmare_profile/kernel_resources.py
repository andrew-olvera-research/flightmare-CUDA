"""Inspect compiled kernels and theoretical occupancy without privileged counters.

Loads extracted cubins from the specified binary; never launches a kernel or
changes device settings. Occupancy is a resource ceiling, not achieved utilization.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def inspect(binary, device=0):
    binary = Path(binary).resolve(strict=True)
    driver = ct.CDLL('libcuda.so.1')
    signatures = {
        'cuInit': [ct.c_uint],
        'cuDeviceGet': [ct.POINTER(ct.c_int), ct.c_int],
        'cuDeviceGetAttribute': [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
        'cuDevicePrimaryCtxRetain': [ct.POINTER(ct.c_void_p), ct.c_int],
        'cuDevicePrimaryCtxRelease_v2': [ct.c_int],
        'cuCtxGetCurrent': [ct.POINTER(ct.c_void_p)],
        'cuCtxSetCurrent': [ct.c_void_p],
        'cuModuleLoad': [ct.POINTER(ct.c_void_p), ct.c_char_p],
        'cuModuleUnload': [ct.c_void_p],
        'cuModuleGetFunction': [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
        'cuFuncGetAttribute': [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
        'cuOccupancyMaxActiveBlocksPerMultiprocessor': [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
    }
    for name, args in signatures.items():
        getattr(driver, name).argtypes = args
        getattr(driver, name).restype = ct.c_int

    def call(name, *args):
        result = getattr(driver, name)(*args)
        if result: raise RuntimeError(f'{name}: CUDA driver error {result}')

    def attribute(name, kind, handle):
        out = ct.c_int()
        call(name, ct.byref(out), kind, handle)
        return out.value

    call('cuInit', 0)
    ordinal, context, previous = ct.c_int(), ct.c_void_p(), ct.c_void_p()
    call('cuDeviceGet', ct.byref(ordinal), device)
    hardware = {name: attribute('cuDeviceGetAttribute', kind, ordinal)
                for name, kind in dict(sm_count=16, warp_size=10, threads_per_sm=39,
                                       registers_per_sm=82, cc_major=75, cc_minor=76).items()}
    report = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  device=device, hardware=hardware, kernels=[],
                  limitation='Theoretical resource occupancy only; no achieved occupancy or DRAM counters')
    call('cuCtxGetCurrent', ct.byref(previous))
    call('cuDevicePrimaryCtxRetain', ct.byref(context), ordinal)
    try:
        call('cuCtxSetCurrent', context)
        with tempfile.TemporaryDirectory(prefix='flightmare-cubins-') as directory:
            subprocess.run(['cuobjdump', '--extract-elf', 'all', str(binary)], cwd=directory,
                           check=True, capture_output=True, text=True)
            cubins = sorted(Path(directory).glob('*.cubin'))
            if not cubins: raise RuntimeError('No compiled CUDA ELF images found')
            for cubin in cubins:
                resource = subprocess.check_output(['cuobjdump','--dump-resource-usage',str(cubin)], text=True)
                symbols = re.findall(r'Function\s+(\S+):', resource)
                module = ct.c_void_p()
                call('cuModuleLoad', ct.byref(module), str(cubin).encode())
                try:
                    for symbol in symbols:
                        function = ct.c_void_p()
                        call('cuModuleGetFunction', ct.byref(function), module, symbol.encode())
                        row = dict(symbol=symbol, cubin=cubin.name,
                                   name=subprocess.check_output(['c++filt',symbol],text=True).strip())
                        row.update({name: attribute('cuFuncGetAttribute', kind, function)
                                    for name, kind in dict(registers=4, shared_bytes=1, local_bytes=3,
                                                           max_threads_per_block=0).items()})
                        row['block_sizes'] = []
                        for threads in (32,64,128,256,512):
                            if threads > row['max_threads_per_block']: continue
                            blocks = ct.c_int()
                            call('cuOccupancyMaxActiveBlocksPerMultiprocessor', ct.byref(blocks), function, threads, 0)
                            row['block_sizes'].append(dict(threads=threads, blocks_per_sm=blocks.value,
                                resident_warps_per_sm=blocks.value*threads//hardware['warp_size'],
                                occupancy_ceiling=blocks.value*threads/hardware['threads_per_sm']))
                        report['kernels'].append(row)
                finally:
                    call('cuModuleUnload', module)
    finally:
        call('cuCtxSetCurrent', previous)
        call('cuDevicePrimaryCtxRelease_v2', ordinal)
    if not report['kernels']: raise RuntimeError('No kernel resources inspected')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.binary, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Inspected {len(report["kernels"])} compiled kernels: {args.output}')


if __name__ == '__main__':
    main()
