import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

vendor = "src/surfdec/_vendor"
sources = ["src/surfdec/_native/decoder.cc"] + sorted(
    f for f in glob.glob(f"{vendor}/pymatching/sparse_blossom/**/*.cc", recursive=True)
)

setup(
    ext_modules=[
        Pybind11Extension(
            "surfdec._native",
            sources,
            include_dirs=[vendor],
            cxx_std=20,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
