#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Performance testing for the maths module.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-11
'''

from timeit import repeat

from numpy import empty
from numpy.random import random

from nerv.maths import cy_softmax
from nerv.maths import cy_tanh_prime
from nerv.maths import py_softmax
from nerv.maths import py_tanh_prime

if __name__ == '__main__':
    dims = 32
    num_its = 10 ** 5
    shape = (dims, 1)
    x = random(shape)
    o = empty(shape)

    # Vector operations.
    for f in (
            py_softmax,
            cy_softmax,
            py_tanh_prime,
            cy_tanh_prime,
            ):
        print(f.__name__, end=' ')
        print(min(repeat(lambda : f(x, out=o), repeat=num_its, number=1)))
