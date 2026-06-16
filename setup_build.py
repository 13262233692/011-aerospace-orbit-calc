import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.fspath(Path(self.get_ext_fullpath(ext.name)).parent.resolve())

        cfg = "Release"

        import pybind11
        pybind11_dir = pybind11.get_cmake_dir()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11_dir}",
        ]
        build_args = ["--config", cfg]

        if sys.platform.startswith("win"):
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
            ]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", "-j4"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )


setup(
    ext_modules=[CMakeExtension("_sgp4_binding", sourcedir="cpp")],
    cmdclass={"build_ext": CMakeBuild},
)
