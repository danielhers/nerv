# vim:set ft=cython ts=4 sw=4 sts=4 autoindent:

'''
Mathematical helpers and short-hands.

Version:    2014-03-11
Author:     Pontus Stenetorp    <pontus stenetorp se>
'''

# TODO: Use memoryviews?
#   http://docs.cython.org/src/userguide/memoryviews.html
# TODO: Use cpdef to simplify interface?
#   http://docs.cython.org/src/reference/language_basics.html#cpdef
# TODO: Use http://docs.cython.org/src/tutorial/pure.html

from numpy import dot
from numpy import empty
from numpy import float64

cimport numpy
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport pow
from libc.string cimport memset
from numpy cimport PyArray_DATA
from numpy cimport float64_t

from cblas cimport CblasRowMajor
from cblas cimport CblasTrans
from cblas cimport cblas_dgemv
from cblas cimport cblas_dger

DOUBLE = float64
ctypedef float64_t DOUBLE_t

cdef inline void _softmax(const int size, const DOUBLE_t *x,
        DOUBLE_t *out) nogil:
    cdef int i
    cdef DOUBLE_t xmax, xsum, tmp

    xmax = x[0]
    for i in range(1, size):
        xmax = max(xmax, x[i])

    xsum = 0
    for i in range(size):
        tmp = exp(x[i] - xmax)
        out[i] = tmp
        xsum += tmp

    for i in range(size):
        out[i] /= xsum

def softmax(x, out=None):
    # XXX: IDENTICAL TO TANH_PRIME! Generate?
    cdef DOUBLE_t *x_ptr, *out_ptr
    cdef int size

    if out is None:
        out = empty(x.shape)

    x_ptr = <DOUBLE_t *> PyArray_DATA(x)
    out_ptr = <DOUBLE_t *> PyArray_DATA(out)
    size = x.size

    _softmax(size, x_ptr, out_ptr)
    return out

# Use?:
#   http://scipy-lectures.github.io/advanced/advanced_numpy/
#       #exercise-building-an-ufunc-from-scratch
cdef inline void _tanh_prime(const int size, const DOUBLE_t *x,
        DOUBLE_t *out) nogil:
    cdef int i

    for i in range(size):
        out[i] = 1 - pow(x[i], 2)

def tanh_prime(x, out=None):
    # XXX: IDENTICAL TO SOFTMAX! Generate?
    cdef DOUBLE_t *x_ptr, *out_ptr
    cdef int size

    if out is None:
        out = empty(x.shape)

    x_ptr = <DOUBLE_t *> PyArray_DATA(x)
    out_ptr = <DOUBLE_t *> PyArray_DATA(out)
    size = x.size

    _tanh_prime(size, x_ptr, out_ptr)
    return out
