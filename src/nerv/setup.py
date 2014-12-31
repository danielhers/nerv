#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

# TODO: Info

# TODO: Move out of the directory?

from Cython.Build import cythonize as cythonise
from distutils.core import Extension
from distutils.core import setup

from numpy import get_include

cy_extensions = cythonise([
        Extension('cy_maths',
            sources=['cy/cy_maths.pyx', ],
            libraries=[
                'blas',
                ],
            ),
        Extension('cy_dag',
            sources=['cy/cy_dag.pyx', ],
            ),
        ])

setup(
    ext_modules=cy_extensions,
    include_dirs=[get_include(), ],
)
