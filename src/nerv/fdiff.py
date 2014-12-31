# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Finite difference and finite difference checks.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-10-28
'''

from copy import deepcopy
from functools import partial
from math import fsum
from sys import stderr

from numpy import allclose as numpy_allclose
from numpy import vstack
from scipy.optimize import approx_fprime

from .maths import EPSILON
from .net import Loss

# Ideally we would have the tolerance depend on the errors we can expect from
#   the finite difference calculations, but there are of course also numerical
#   errors involved since we have plenty of floating point calculations taking
#   place within the net.
allclose = partial(numpy_allclose, atol=1e-6)

def fdiff(m, f):
    return approx_fprime(m, f, EPSILON)

def loss_given_m(model, nets, model_key, bias=False):
    def f(m):
        _model = deepcopy(model)
        if not bias:
            w_dic = _model.weight
        else:
            w_dic = _model.bias
        w = w_dic[model_key]

        if isinstance(w, dict):
            offset = 0
            for _w in w.values():
                _w[:] = m[offset:offset + _w.size].reshape(_w.shape)
                offset += _w.size
        if isinstance(w, tuple):
            offset = 0
            for _w in w:
                _w[:] = m[offset:offset + _w.size].reshape(_w.shape)
                offset += _w.size
        else:
            w[:] = m.reshape(w.shape)

        loss = Loss()
        for net in nets:
            net.forward(_model, loss=loss)
        return loss.total()
    return f

# TODO: Throw an exception instead?
def fdiff_check(model, net, check_bias=True, verbose=False):
    loss = Loss()
    net.forward(model, loss=loss)
    gradient = model.gradient()
    net.backward(model, gradient=gradient)

    loss_f_gen = partial(loss_given_m, model, (net, ))

    for vertice_class in model.vertice_classes:
        for bias in (False, ) if not check_bias else (False, True, ):
            key = vertice_class.name
            if not bias:
                m_dic = model.weight
                g_dic = gradient.weight
            else:
                m_dic = model.bias
                g_dic = gradient.bias

            c = m_dic[key]
            if isinstance(c, dict):
                m = vstack(c.values())
                g = vstack(g_dic[key].values())
            elif isinstance(c, tuple):
                from numpy import hstack
                m = hstack(e.flatten() for e in c)
                g = hstack(e.flatten() for e in g_dic[key])
            else:
                # "Standard".
                m = c
                g = g_dic[key]

            _n = fdiff(m.flatten(),
                    loss_f_gen(key, bias=bias))
            n = _n.reshape(g.shape)

            mismatch = not allclose(g, n)
            if verbose:
                print(vertice_class.__name__, '' if not bias else 'bias',
                        file=stderr)
            if mismatch:
                print(('WARNING: Mismatch detected for the {} '
                    '{}gradient.').format(vertice_class.__name__,
                        '' if not bias else 'bias '), file=stderr)
            if mismatch or verbose:
                print('\tAnalytical:', '\t'.join(str(e)
                    for e in g.flatten()), file=stderr)
                print('\t Numerical:', '\t'.join(str(e)
                    for e in n.flatten()), file=stderr)
                print(file=stderr)
