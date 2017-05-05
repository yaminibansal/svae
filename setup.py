from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(
    name='svae',
    packages=['svae', 'svae.models', 'svae.hmm', 'svae.distributions']
    #ext_modules=cythonize('**/*.pyx'),
    #include_dirs=[np.get_include(),],
)
