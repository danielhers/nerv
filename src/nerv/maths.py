# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Mathematical helpers and short-hands.

Version:    2013-02-06
Author:     Pontus Stenetorp    <pontus stenetorp se>
'''

from numpy import add
from numpy import array
from numpy import copyto
from numpy import dot
from numpy import empty
from numpy import exp
from numpy import finfo
from numpy import log
from numpy import negative
from numpy import power
from numpy import sqrt
from numpy import tanh

EPSILON = sqrt(finfo(float).eps)

def py_softmax(x, out=None):
    if out is None:
        out = array(x, copy=True)
    else:
        copyto(out, x)
    out -= out.max()
    exp(out, out=out)
    out /= out.sum()
    return out

try:
    from .cy_maths import softmax as cy_softmax
    softmax = cy_softmax
except ImportError:
    def cy_softmax(x, out=None):
        raise NotImplementedError
    softmax = py_softmax

# TODO: Could consider a Cythonised version, but it shouldn't really help.
tanh = tanh

def py_tanh_prime(x, out=None):
    if out is None:
        out = empty(x.shape)
    out[:] = 1 - x ** 2
    return out

try:
    from .cy_maths import tanh_prime as cy_tanh_prime
    tanh_prime = cy_tanh_prime
except ImportError:
    def cy_tanh_prime(x, out=None):
        raise NotImplementedError
    tanh_prime = py_tanh_prime
