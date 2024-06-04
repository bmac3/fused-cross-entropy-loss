from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "fused_ce.c_ext",
        ["fused_ce/fused_ce.cpp"],
    ),
]

setup(
    name='fused-ce',
    version='0.0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
