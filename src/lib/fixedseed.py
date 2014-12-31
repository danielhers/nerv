#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Perform operations with a fixed seed for a block, then restore the state of
the PRNG. All using the with syntax:

    with FixedSeed(0x4711):
        ...

Has this already been done? It is quite handy for experiments since it makes
sure that you don't forget to restore the state.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-24
'''

from random import getstate
from random import seed
from random import setstate

try:
    from numpy.random import get_state as np_getstate
    from numpy.random import seed as np_seed
    from numpy.random import set_state as np_setstate
    NUMPY = True
except ImportError:
    NUMPY = False


class FixedSeed(object):
    def __init__(self, seed):
        self.seed = seed
        self._state = None
        self._np_state = None

    def __enter__(self):
        self._state = getstate()
        if NUMPY:
            self._np_state = np_getstate()
        seed(self.seed)
        if NUMPY:
            np_seed(self.seed)
        return self

    def __exit__(self, type, value, traceback):
        setstate(self._state)
        if NUMPY:
            np_setstate(self._np_state)


# Some minor testing.
if __name__ == '__main__':
    from sys import stderr

    def test(numpy=False):
        if not numpy:
            from random import random
        else:
            from numpy.random import random

        with FixedSeed(0x4711):
            a = [random() for _ in range(0x17)]
        after_a = random()

        with FixedSeed(0x4711):
            b = [random() for _ in range(0x17)]
        after_b = random()

        assert a == b
        # Should fail in most cases
        assert after_a != after_b

    # Test the standard Python implementation
    test()

    # Test the numpy implementation
    try:
        test(numpy=True)
    except ImportError:
        print('WARNING: numpy import failure, skipping numpy tests',
                file=stderr)
