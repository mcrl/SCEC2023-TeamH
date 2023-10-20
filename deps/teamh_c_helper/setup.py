from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from torch.utils.cpp_extension import CUDAExtension

ext_modules = [
    CUDAExtension(
        name = "teamh_c_helper",
        sources = sorted(glob("csrc/*.cpp")),  # Sort source files for reproducibility
        undef_macros=['NDEBUG'],
    ),
]

setup(
  name="teamh_c_helper",
  ext_modules=ext_modules
)