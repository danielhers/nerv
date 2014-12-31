#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Performance testing for the net module.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-12
'''

from collections import OrderedDict
from functools import partial
from itertools import chain
from itertools import islice
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from random import randint
from sys import stderr
from time import time

from numpy import zeros

from lib.fixedseed import FixedSeed

from nerv.init import random_uniform
from nerv.lang import sentgen
from nerv.net import Net
from nerv.net import keyed_source_vertex
from nerv.net import net_model
from nerv.net import net_model
from nerv.net import rnn_vertex
from nerv.net import softmax_vertex
from nerv.rand import decorate
from nerv.rand import onehot
from nerv.rand import bintree

### Constants
NUM_THROUGHS = 4
NUM_THROUGH_NETS = 32
COMP_VERT_C_F = (
    rnn_vertex,
    )
STRUCT_F = (
        bintree,
        )
###

def _rand_class(class_class, lbls):
    target = zeros(lbls).reshape(-1, 1)
    target[randint(0, lbls - 1)] = 1
    return class_class(target=target)

def _gen(n, comp_c_f, struct_f=bintree):
    dims = 32
    lbls = 5
    unk = '<UNK>'

    sents = tuple(tuple(s) for s in islice(sentgen(min_len=2), n))
    vocab = set(chain.from_iterable(sents))

    dic = OrderedDict()
    for tok in vocab:
        dic[tok] = random_uniform(dims)
    dic[unk] = random_uniform(dims)

    Source = keyed_source_vertex(dims, dic, missing_=unk)
    Class = softmax_vertex(lbls, dims)
    Comp = comp_c_f(dims, 2)
    Model = net_model((Source, Comp, Class, ))

    def class_c():
        return Class(target=onehot(lbls))

    nets = []
    for sent in sents:
        net = struct_f(Comp, (Source(tok) for tok in sent))
        decorate(net, class_c, internal_prob=0.2)
        nets.append(net)

    return (tuple(nets), Model, Comp)

# Hot-spot profiling.
def _perf_hot_spot():
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print('WARNING: Unable to import line_profiler, skipping '
                'detailed performance checking', file=stderr)
        return

    for comp_c_f in COMP_VERT_C_F:
        # Generate a random net.
        with FixedSeed(0x4711):
            nets, Model, Comp = _gen(1, comp_c_f)
            net = nets[0]

        model = Model()

        # Attach the profiler.
        profiler = LineProfiler()
        profiler.add_function(Comp.forward)
        profiler.add_function(Comp.backward)
        profiler.enable()

        # Run the network forwards and backwards to accumulate statistics.
        gradient = model.gradient()
        for _ in range(32):
            net.forward(model)
            net.backward(model, gradient=gradient)

        profiler.print_stats()

        sanity = False
        if sanity:
            from .fdiff import fdiff_check
            fdiff_check(model, net)

def _perf_through():
    for struct_f in STRUCT_F:
        print('{}:'.format(struct_f.__name__))
        for comp_c_f in COMP_VERT_C_F:
            # Generate the randoms nets and models.
            with FixedSeed(0x17):
                nets, Model, _ = _gen(NUM_THROUGH_NETS, comp_c_f,
                        struct_f=struct_f)
                model = Model()

            tocs = []
            for _ in range(NUM_THROUGHS):
                tic = time()
                for net in nets:
                    net.forward(model)
                tocs.append(time() - tic)
            print(('{} sequential forward throughput: {:.1f} net(s) per second'
                ).format(comp_c_f.__name__, NUM_THROUGH_NETS / min(*tocs)))

            tocs = []
            for _ in range(NUM_THROUGHS):
                gradient = model.gradient()
                tic = time()
                for net in nets:
                    net.backward(model, gradient=gradient)
                tocs.append(time() - tic)
            print(('{} sequential backward throughput: {:.1f} net(s) per '
                'second').format(comp_c_f.__name__,
                    NUM_THROUGH_NETS / min(*tocs)))

def _perf_through_par():
    num_threads = min(cpu_count(), 4)
    pool = ThreadPool(num_threads)

    for comp_c_f in COMP_VERT_C_F:
        # Generate the randoms nets and models.
        with FixedSeed(0x17):
            nets, Model, _ = _gen(NUM_THROUGH_NETS, comp_c_f)
            model = Model()


        tocs = []
        for _ in range(NUM_THROUGHS):
            tic = time()
            for _ in pool.imap_unordered(model.forward, nets):
                pass
            tocs.append(time() - tic)
        print(('{} parallel ({} thread(s)) forward throughput: {:.1f} net(s) '
            'per second').format(comp_c_f.__name__, num_threads,
                NUM_THROUGH_NETS / min(*tocs)))

        # TODO: Can't share gradient? Needs to be fixed?
        tocs = []
        for _ in range(NUM_THROUGHS):
            tic = time()
            for _ in pool.imap_unordered(model.backward, nets):
                pass
            tocs.append(time() - tic)
        print(('{} parallel ({} thread(s)) backward throughput: {:.1f} '
            'net(s) per second').format(comp_c_f.__name__, num_threads,
                NUM_THROUGH_NETS / min(*tocs)))

if __name__ == '__main__':
    # Fix the seeds for more consistent results.
    with FixedSeed(0x4711):
        _perf_hot_spot()
    with FixedSeed(0x4711):
        _perf_through()
    with FixedSeed(0x4711):
        _perf_through_par()
