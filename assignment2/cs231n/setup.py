from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
# import os

# path = os.path.join(os.pardir, 'im2col_cython')
extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
