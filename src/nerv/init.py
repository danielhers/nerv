# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Initialisation functions for networks and representations.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-01-29
'''

# XXX: We need to re-think how we handle shape vs. dimension/numparents.

from collections import defaultdict
from functools import partial
from itertools import count
from math import sqrt

from numpy import array
from numpy import empty
from numpy.linalg import norm
from numpy.random import normal
from numpy.random import rand
from numpy.random import randn
from numpy.random import random

# Normalised initialisation heuristic from Glorot and Bengio (2010).
def init_range(fan_out, fan_in=1, sigmoid=False):
    upper = sqrt(6 / (fan_in + fan_out)) * (4 if sigmoid else 1)
    return (-upper, upper, )

def init_layer(shape, fan_out, fan_in=1, sigmoid=False):
    _, upper = init_range(fan_out, fan_in=fan_in, sigmoid=sigmoid)
    return rand(*shape) * upper * 2 - upper

def random_uniform(dimensionality, lower=-1.0, upper=1.0):
    assert upper >= lower
    width = upper - lower
    return random((dimensionality, 1)) * width + lower

# Default to uniform [-1.0, 1.0] a'la Collobert and Weston
# XXX: Still need to verify if this really is the case.
def uniform_random_reprs(dimensionality, lower=-1.0, upper=1.0):
    return defaultdict(partial(random_uniform, dimensionality, lower=lower,
        upper=upper))

def random_unitvec(dimensionality):
    vec = normal(loc=0, scale=1, size=(dimensionality, 1, ))
    return vec / norm(vec)

def unitvec_random_reprs(dimensionality):
    return defaultdict(partial(random_unitvec, dimensionality))

# Default to 0 mean and 0.1 standard deviation as in "Recursive Deep Models
#   for Semantic Compositionality Over a Sentiment Treebank"
#   by Socher et al. (2013). Reversed from the Matlab code (yes, this
#   disagrees with what the paper says).
def gaussian_repr(dimensionality, mean=0, sigma=0.1):
    return normal(loc=mean, scale=sigma, size=(dimensionality, 1, ))

def gaussian_random_reprs(dimensionality, mean=0, sigma=0.1):
    return defaultdict(partial(gaussian_repr, dimensionality, mean=mean,
        sigma=sigma))

# Block identity matrices and a uniform [-0.001,0.001] distribution added.
#   See "Parsing with Compositional Vector Grammars" by Socher et al. (2013).
def socher_2013_comp_mtrx(dimensionality, num_parents):
    mtrx = randn(dimensionality, num_parents * dimensionality) / 100
    for i in range(mtrx.shape[1]):
        mtrx[i % mtrx.shape[0]][i] += 1 / num_parents
    return mtrx

# Similar to Socher et al. (2013) but decrases the diagonal as the distance
#   to each child increases (assuming distant words having less influence).
# Note: It bothers me a bit that it doesn't sum to one though.
def gradual_comp_mtrx(dimensionality, num_parents):
    mtrx = randn(dimensionality, num_parents * dimensionality) / 100
    for i in range(mtrx.shape[1]):
        j = i % mtrx.shape[0]
        factor = i // (2 * dimensionality) * 2 + 2
        mtrx[j][i] += 1 / factor
    return mtrx

def onehot_reprs(dimensionality, hot=1.0, cold=0.0):
    idx_it = count()
    def next_vec():
        curr_hot = next(idx_it)
        if curr_hot < dimensionality:
            return array([hot if i == curr_hot else cold
                for i in range(dimensionality)])
        else:
            assert False, ('unable to assign more than {} one-hot '
                    'representations for a vector of size {}'
                    ).format(dimensionality, dimensionality)
    return defaultdict(next_vec)
