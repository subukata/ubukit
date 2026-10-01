"""Optional build: .venv/bin/python som_candidate/setup_native.py build_ext --inplace."""
from pathlib import Path
import os
import numpy as np
from setuptools import Extension,setup
from Cython.Build import cythonize
os.environ.setdefault('CC','gcc');os.environ.setdefault('CXX','g++')
here=Path(__file__).resolve().parent;os.chdir(here.parent)
ext=Extension('som_candidate._native_v2',[str(here/'native_src/_kernels_v2.pyx')],
    include_dirs=[np.get_include(),str(here/'native_src')],
    depends=[str(here/'native_src/native_v2.h')],
    define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
    extra_compile_args=['-O3','-march=native','-fopenmp','-fno-math-errno','-ffp-contract=off'],
    extra_link_args=['-fopenmp'],libraries=['m','mvec'])
setup(name='som-candidate-native-restart-v2',ext_modules=cythonize([ext],compiler_directives={'language_level':3}))
