#!/usr/bin/python3

__author__ = 'danielh'

import argparse
import sys
import operator
from collections import OrderedDict, defaultdict
from numpy import array

from ucca import layer0, layer1, core
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
    # units = [unit for unit in passage.layer(layer1.LAYER_ID).all if not unit.parents]
    # while units:
    #     next_units = []
    #     for unit in units:
            # assert len(unit.children) in (0, 2),\
            #     "passage must be binarized, but unit has %d child%s: %s" %\
            #     (len(unit.children), "" if len(unit.children) == 1 else "ren", unit)
            # for child in unit.children:
            #     next_units.append(child)
            #     net.add_edge(vertices[unit.ID], vertices[child.ID])
        # units = next_units

    model = Model()
    model.forward(net)
    print(labels["1.1"].activations)

    return model

def topological_sort(passage):
    # sort into topological ordering to create parents before children
    levels = defaultdict(set)   # levels start from 0 (root)
    level_by_id = {}
    remaining = passage.layer(layer0.LAYER_ID).all  # terminals: leaves
    while remaining:
        unit = remaining.pop()
        if unit.ID in level_by_id:  # done already
            pass
        elif not unit.parents:  # root
            level_by_id[unit.ID] = 0
            levels[0].add(unit)
        elif not all(parent.ID in level_by_id for parent in unit.parents):
            remaining.append(unit)
            remaining.extend(parent for parent in unit.parents if parent.ID not in level_by_id)
        else:  # done with parents
            level = 1 + max(level_by_id[parent.ID] for parent in unit.parents)
            level_by_id[unit.ID] = level
            levels[level].add(unit)

    return [unit for level, level_nodes in sorted(levels.items())
            for unit in sorted(level_nodes,
                               key=lambda x: int(x.ID.split(core.Node.ID_SEPARATOR)[1]))]

def binarize(passage):
    p = core.Passage(passage.ID)
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    old_id_to_new = {}

    linkages = []
    for unit in topological_sort(passage):
        if unit.tag == layer1.NodeTags.Linkage:
            linkages.append(unit)
        elif not unit.parents:
            old_id_to_new[unit.ID] = None
        elif unit.ID in old_id_to_new:
            pass
        elif unit.layer.ID == layer0.LAYER_ID:
            old_id_to_new[unit.ID] =\
                l0.add_terminal(text=unit.text, punct=unit.punct, paragraph=unit.paragraph)
        elif len(unit.children) in (0, 2):
                node = l1.add_fnode(old_id_to_new[unit.parents[0].ID],
                                    unit.incoming[0].tag)
                for edge in unit.incoming[1:]:
                    if edge.tag not in (layer1.EdgeTags.Linker,
                                        layer1.EdgeTags.LinkArgument,
                                        layer1.EdgeTags.LinkRelation):
                        old_id_to_new[edge.parent.ID].add(node, edge.tag)
                old_id_to_new[unit.ID] = node
        elif len(unit.children) == 1:  # single child, collapse
            child_edge = unit.outgoing[0]
            node = l1.add_fnode(old_id_to_new[unit.parents[0].ID],
                                unit.incoming[0].tag + "$" + child_edge.tag)
            for edge in unit.incoming[1:]:
                if edge.tag not in (layer1.EdgeTags.Linker,
                                    layer1.EdgeTags.LinkArgument,
                                    layer1.EdgeTags.LinkRelation):
                    old_id_to_new[edge.parent.ID].add(edge.tag + "$" + child_edge.tag, node)
        else:  # binarize
            node = l1.add_fnode(old_id_to_new[unit.parents[0].ID],
                                unit.incoming[0].tag)
            old_id_to_new[unit.ID] = node
            for edge in unit.incoming[1:]:
                if edge.tag not in (layer1.EdgeTags.Linker,
                                    layer1.EdgeTags.LinkArgument,
                                    layer1.EdgeTags.LinkRelation):
                    old_id_to_new[edge.parent.ID].add(node, edge.tag)
            other_edges = unit.outgoing[1:]
            edge_tags = [edge.tag for edge in other_edges]
            new = l1.add_fnode(node, '#'.join(edge_tags))
            for edge in other_edges:
                if edge.tag not in (layer1.EdgeTags.Linker,
                                    layer1.EdgeTags.LinkArgument,
                                    layer1.EdgeTags.LinkRelation):
                    if unit.ID in old_id_to_new:
                        pass
                    elif unit.layer.ID == layer0.LAYER_ID:
                        old_id_to_new[edge.child.ID] =\
                            l0.add_terminal(text=edge.child.ID.text,
                                            punct=edge.child.ID.punct,
                                            paragraph=edge.child.ID.paragraph)
                    else:
                        old_id_to_new[edge.child.ID] = l1.add_fnode(new, edge.tag)

    for linkage in linkages:
        pass
        # l1.add_linkage(linkage.children[0], *linkage.children[1:])

    return p

def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="passage file names to convert")
    args = parser.parse_args()

    for filename in args.filenames:
        passage = file2passage(filename)
        # print("original: %s" % passage.layer(layer1.LAYER_ID).heads[0])
        # p = binarize(passage)
        # print("binarized: %s" % p.layer(layer1.LAYER_ID).heads[0])
        output = create_model(passage)

    sys.exit(0)


if __name__ == '__main__':
    main()