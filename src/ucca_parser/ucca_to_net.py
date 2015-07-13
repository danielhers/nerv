#!/usr/bin/python3

__author__ = 'danielh'

import argparse
import sys
import operator
from collections import OrderedDict
from numpy import array

from ucca import layer0, layer1
from util import file2passage
from nerv.net import keyed_source_vertex
from nerv.net import average_vertex
from nerv.net import softmax_vertex
from nerv.net import net_model
from nerv.net import Net
from nerv.init import random_uniform

desc = """Parses an XML in UCCA standard format, and creates a nerv DAG net model.
"""

# The dimensionality of our token representations.
dims = 32

def create_model(passage):
    terminals = sorted(passage.layer(layer0.LAYER_ID).all,
                       key=operator.attrgetter('position'))
    non_terminals = passage.layer(layer1.LAYER_ID).all
    # Mappings between each token and a randomly initialised representation.
    reprs = OrderedDict([(t.text, random_uniform(dims)) for t in terminals])
    reprs['<unk>'] = random_uniform(dims)

    # One-hot labels as column vectors.
    neg = array((1, 0)).reshape(-1, 1)
    pos = array((0, 1)).reshape(-1, 1)
    labels = (neg, pos)

    # Generate classes for each desired vertex.
    TerminalVertex = keyed_source_vertex(dims, reprs)
    # Composes two vertices of a desired dimensionality.
    NonTerminalVertex = average_vertex(dims)
    # Predict one out of several labels.
    LabelVertex = softmax_vertex(len(labels), dims)
    # Generate a model class that is applicable to our vertices.
    Model = net_model((TerminalVertex, NonTerminalVertex, LabelVertex))
    # Let us now construct a net for our example sentence.
    net = Net()

    vertices = dict([(t.ID, TerminalVertex(t.text)) for t in terminals] +
                    [(u.ID, NonTerminalVertex()) for u in non_terminals])
    labels = dict([(u.ID, LabelVertex(pos)) for u in non_terminals])
    for unit in non_terminals:
        net.add_edge(vertices[unit.ID], labels[unit.ID])
        for child in unit.children:
            net.add_edge(vertices[child.ID], vertices[unit.ID])

    model = Model()
    model.forward(net)
    print(labels["1.1"].activations)

    return model


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="passage file names to convert")
    args = parser.parse_args()

    for filename in args.filenames:
        passage = file2passage(filename)
        output = create_model(passage)

    sys.exit(0)


if __name__ == '__main__':
    main()