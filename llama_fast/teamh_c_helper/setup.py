from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "teamh_c_helper",
        sorted(glob("csrc/*.cpp")),  # Sort source files for reproducibility
    ),
]

setup(
  name="teamh_c_helper",
  ext_modules=ext_modules
)