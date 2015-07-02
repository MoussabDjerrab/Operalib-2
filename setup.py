from setuptools import find_packages
from distutils.core import setup
from Cython.Build import cythonize

import os
import numpy
import OVFM

def read( fname ):
    return open( os.path.join( os.path.dirname( __file__ ), fname ) ).read( )

setup(
    name = "OperaLib",
    version = OVFM.__version__,
    ext_modules = cythonize( [ "operalib/*.pyx"], include_path = [ "operalib" ], language="C" ),
    include_dirs = [ numpy.get_include( ) ],
    packages = find_packages( ),
    include_package_data = True,
    author = "Romain Brault",
    author_email = "romain.brault@telecom-paris.fr",
    description = ("Learing with operator-valued kernels"),
    license = "MIT",
    keywords = "operator-valued kernels",
    url = "",
    long_description = read('README.md'),
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 1 - Alpha",
        "Topic :: Machine learning",
        "Programming Language :: Python :: 2.7",
        "License :: MIT License",
    ],
)
