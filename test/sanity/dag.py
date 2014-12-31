#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Sanity checking for the dag module.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-04-29
'''

from nerv.dag import OrderedSet

from random import shuffle

if __name__ == '__main__':
    def ordered_set():
        seq = list(range(7))
        shuffle(seq)
        seq = tuple(seq)

        oset = OrderedSet()

        for item in seq:
            oset.add(item)

        assert oset

        for item in seq:
            assert item in oset

        for oset_item, seq_item in zip(oset, seq):
            assert oset_item == seq_item

        for item in seq:
            oset.discard(item)

        for item in seq:
            assert item not in oset

        assert not oset

    ordered_set()
