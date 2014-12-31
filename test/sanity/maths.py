#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Sanity testing for the maths module.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-11
'''

# TODO: Not testing "tricky" ranges.

from sys import stderr

from numpy import allclose
from numpy.random import random

from nerv.maths import cy_softmax
from nerv.maths import cy_tanh_prime
from nerv.maths import py_softmax
from nerv.maths import py_tanh_prime

if __name__ == '__main__':
    dims = 2 # XXX: 32

    # Vector operations.
    for f_ref, f in (
            (py_softmax, cy_softmax),
            (py_tanh_prime, cy_tanh_prime),
            ):
        shape = (dims, 1)
        x = random(shape)
        o_ref = f_ref(x)
        try:
            o = f(x)
        except NotImplementedError:
            continue

        if not allclose(o_ref, o):
            print(('WARNING: Sanity check failed with reference {} for '
                '{}').format(f_ref.__name__, f.__name__), file=stderr)
            print('Reference:', o_ref.T, file=stderr)
            print('Output:   ', o.T, file=stderr)
