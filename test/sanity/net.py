#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Sanity testing for the net module.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-11
'''

# TODO: This module could be cleaned up.

from pickle import dumps
from random import randint
from sys import stderr

from lib.fixedseed import FixedSeed

from nerv.init import random_uniform
from nerv.net import Net
from nerv.net import keyed_source_vertex
from nerv.net import net_model
from nerv.net import rnn_vertex
from nerv.net import softmax_vertex
from nerv.net import static_source_vertex

# Classification vertex with a random target.
def _rand_class(class_class, lbls):
    target = zeros(lbls).reshape(-1, 1)
    target[randint(0, lbls - 1)] = 1
    return class_class(target=target)

if __name__ == '__main__':
    def gradient_check():
        from collections import OrderedDict

        from numpy import array

        from nerv.fdiff import fdiff_check

        def softmax():
            dims = 2
            lbls = 2

            Source = static_source_vertex(dims)
            Class = softmax_vertex(lbls, 2 * dims)
            Model = net_model((Class, ))

            a = Source(random_uniform(dims))
            b = Source(random_uniform(dims))
            c = Class(target=array((1, 0, )).reshape(-1, 1))

            net = Net()
            net.add_edge(a, c)
            net.add_edge(b, c)

            model = Model()

            return (model, net, )

        def keyed():
            dims = 2
            lbls = 2

            dic = OrderedDict((
                    ('a', random_uniform(dims), ),
                    ('b', random_uniform(dims), ),
                    ('<UNK>', random_uniform(dims), ),
                    ))

            Source = keyed_source_vertex(dims, dic, missing_='<UNK>')
            Class = softmax_vertex(lbls, 3 * dims)
            Model = net_model((Source, Class, ))

            a = Source('a')
            b = Source('b')
            c = Source('foobar')
            d = Class(target=array((1, 0, )).reshape(-1, 1))

            net = Net()
            net.add_edge(a, d)
            net.add_edge(b, d)
            net.add_edge(c, d)

            model = Model()

            return (model, net, )

        def rnn():
            dims = 2
            lbls = 2

            dic = OrderedDict((
                    ('a', random_uniform(dims), ),
                    ('b', random_uniform(dims), ),
                    ('<UNK>', random_uniform(dims), ),
                    ))

            Source = keyed_source_vertex(dims, dic, missing_='<UNK>')
            Comp = rnn_vertex(dims, 2)
            Class = softmax_vertex(lbls, dims)
            Model = net_model((Source, Comp, Class, ))

            a = Source('a')
            b = Source('b')
            c = Comp()
            d = Class(target=array((1, 0, )).reshape(-1, 1))

            net = Net()
            net.add_edge(a, c)
            net.add_edge(b, c)
            net.add_edge(c, d)

            model = Model()

            return (model, net, )

        for data_f in (
                softmax,
                keyed,
                rnn,
                ):
            fdiff_check(*data_f())

    def pickle_check():
        dims = 32

        vertex_classes = (
                static_source_vertex(dims),
                # Note: This is not how the KeyedSourceVertex is used.
                keyed_source_vertex(dims, {'a': 'foo'}),
                softmax_vertex(dims, dims),
                rnn_vertex(dims, 2),
                )

        for vertex_class in vertex_classes:
            dumps(vertex_class)

        Model = net_model(vertex_classes)

        dumps(Model)

    # Run the actual tests.
    with FixedSeed(0x4711):
        gradient_check()

    pickle_check()
