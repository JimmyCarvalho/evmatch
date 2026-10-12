import glob
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

vendor = "csrc/vendor"
sources = ["csrc/decoder.cc"] + sorted(
    f for f in glob.glob(f"{vendor}/pymatching/sparse_blossom/**/*.cc", recursive=True)
)

setup(
    ext_modules=[
        Pybind11Extension(
            "evmatch._native",
            sources,
            include_dirs=[vendor],
            cxx_std=20,
            extra_compile_args=[] if sys.platform == "win32" else ["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
