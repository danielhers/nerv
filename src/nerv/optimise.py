#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
A collection of function optimisation methods.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-06-12
'''

# TODO: Clean-up!

from itertools import repeat
from multiprocessing import Array
from multiprocessing import Value
from multiprocessing import Process

from numpy import absolute
from numpy import empty
from numpy import frombuffer
from numpy import nonzero
from numpy import sqrt
from numpy import zeros

# Convenience wrapper to support fprime as a joint/separate argument.
def _f(func, fprime):
    if fprime is None:
        return func
    else:
        def f(x):
            return (func(x), fprime(x), )
        return f

# RMSProp from "Lecture 6.5 - rmsprop" by Tieleman and Hinton (2012), yes...
#   that is the actual cite.
#
def fmin_rmsprop(func, x0, fprime=None, learning_rate=0.001, decay_rate=0.1,
        mean_square=None, epsilon=10 ** -7):
    if mean_square is None:
        mean_square = zeros(x0.shape)

    f = _f(func, fprime)

    while True:
        loss, grad = f(x0)

        # Support growing the weight vector/gradient on-the-fly.
        if mean_square.shape != grad.shape:
            mean_square.resize(grad.shape, refcheck=False)

        mean_square[:] = ((1 - decay_rate) * mean_square
                + decay_rate * grad ** 2)
        update = learning_rate * grad
        update[:] /= sqrt(mean_square) + epsilon
        x0[:] -= update

        yield (x0, loss, mean_square)

# AdaGrad, diagonal version, from "Adaptive Subgradient Methods for Online
#   Learning and Stochastic Optimization" by Duchi et al. (2011).
#
# TODO: How much do we need to support? Steal some of it from SciPy with their wrappers?
# TODO: Motivate the choice of constants for AdaGrad?
# TODO: Rip out the "adagrad" prefixes.
# TODO: Use some sort of convergence criterion?
# TODO: Use an "iterate" flag to control behaviour?
# TODO: Make sure we are in line with the paper (I think we are).
def fmin_adagrad(func, x0, fprime=None, learning_rate=0.1,
        sum_grad_square=None, epsilon=1e-3):

    # TODO: We allow you to pass this on to resume a previous call.
    if sum_grad_square is None:
        # The sum of squared gradient components
        sum_grad_square = zeros(x0.shape)

    f = _f(func, fprime)

    while True:
        loss, gradient = f(x0)

        # Support growing the weight vector/gradient on-the-fly.
        if sum_grad_square.shape != gradient.shape:
            sum_grad_square.resize(gradient.shape, refcheck=False)

        sum_grad_square[:] += gradient ** 2
        x0[:] -= (learning_rate * gradient  / (
            sqrt(sum_grad_square) + epsilon))

        yield x0, loss, sum_grad_square

# Stochastic gradient descent with momentum (Polyak, 1964).
def fmin_sgd(func, x0, fprime=None, learning_rate=0.01, momentum_coeff=0.5):
    momentum = zeros(x0.shape)

    f = _f(func, fprime)

    while True:
        loss, gradient = f(x0)
        momentum[:] = momentum * momentum_coeff - gradient * learning_rate
        x0[:] += momentum
        yield (x0, loss, )

# Nestorov's Accelerated Gradient (Nestorov, 1983) in its momentum formulation
#   by Sutskever et al. (2013).
# Note: We do not implement the momentum coefficient schedule described
#   by Sutskever et al. (2013).
def fmin_nag(func, x0, fprime=None, learning_rate=0.01, momentum_coeff=0.5):
    momentum = zeros(x0.shape)

    f = _f(func, fprime)

    while True:
        curr_momentum = momentum_coeff * momentum
        loss, gradient = f(x0 + curr_momentum)
        momentum[:] = curr_momentum - gradient * learning_rate
        x0[:] += momentum
        yield (x0, loss, )

# TODO: Move to sanity!
if __name__ == '__main__':
    from numpy import array
    from numpy.random import uniform

    def func(x):
        return x ** 2, 2 * x

    x0_start = array(tuple(uniform(-(2 ** 3), 2 ** 3)
        for _ in range(2 ** 3)))

    for fmin_f in (
            fmin_sgd,
            fmin_rmsprop,
            fmin_adagrad,
            fmin_nag,
            ):
        print('Testing:', fmin_f.__name__)

        x0 = x0_start.copy()
        print('\tInitial weights:', x0)
        iteration = 0
        for step in fmin_f(func, x0):
            curr_x0, loss = (step[0], step[1], )
            if iteration % 1000 == 0:
                print('\t\tLoss at iteration {}: {}'.format(iteration,
                    sum(loss)))
            if iteration > 0 and iteration % 10000 == 0:
                break
            iteration += 1
        print('\tFinal loss after {} iterations: {}'.format(iteration,
            sum(loss)))
        print('\tFinal weights:', curr_x0)
