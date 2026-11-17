import os
from pathlib import Path
import subprocess
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        dest=Path(self.get_ext_fullpath(ext.name)).resolve()
        build=Path(self.build_temp).resolve(); build.mkdir(parents=True,exist_ok=True)
        subprocess.check_call(['cmake','-S',str(Path(__file__).parent.resolve()),'-B',str(build),
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY='+str(dest.parent),
            '-DRL_EXTENSION_NAME='+dest.name,
            '-DCMAKE_CUDA_ARCHITECTURES='+os.environ.get('FLIGHTMARE_CUDA_ARCHITECTURES','native')])
        subprocess.check_call(['cmake','--build',str(build),'--parallel','4'])


setup(ext_modules=[Extension('flightrl_cuda._kernels',sources=[])],cmdclass={'build_ext':CMakeBuild})
