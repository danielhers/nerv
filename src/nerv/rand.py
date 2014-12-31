# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Pseudo-random structures.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-12
'''

from random import randint

from numpy import zeros
from numpy.random import random

from .dag import VertexType
from .net import Net

def onehot(dims):
    x = zeros((dims, 1))
    x[randint(0, dims - 1)] = 1
    return x

# TODO: Not actually random.
def bintree(comp_c, leaves):
    net = Net()
    vertices = list(leaves)
    while len(vertices) > 1:
        i = randint(0, len(vertices) - 2)
        left = vertices.pop(i)
        right = vertices.pop(i)
        parent = comp_c()
        net.add_edge(left, parent)
        net.add_edge(right, parent)
        vertices.append(parent)
    return net

# Assign children to sources/sinks and with a probablity to internal vertices
#   in an existing net.
def decorate(net, child_c, internal_prob=1.0):
    for v_type, vertex in tuple(net.typed_it()):
        if v_type == VertexType.SOURCE or v_type == VertexType.SINK:
            net.add_edge(vertex, child_c())
        else:
            if random() >= (1 - internal_prob):
                net.add_edge(vertex, child_c())
