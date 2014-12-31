# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
A collection of useful custom data structures.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-02-13
'''

from collections import MutableSet
from collections import defaultdict

# Note: Since Python 3.4+ isn't wide-spread yet in early 2014.
# Adapted from: http://stackoverflow.com/questions/36932/
def _enum(enum_dic):
    _reverse = {v: k for k, v in enum_dic.items()}


    class Enum(object):
        def enums(self):
            return enum_dic.items()

        def __len__(self):
            return len(enum_dic)

        def reverse(cls, key):
            return _reverse[key]


    for name, value in enum_dic.items():
        setattr(Enum, name, value)


    return Enum

def enum(*sequential, **named):
    enum_dic = dict(zip(sequential, range(len(sequential))), **named)
    return _enum(enum_dic)()


# Default dict that gives the missing key to the default_factor.
class KeyedDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        return self.setdefault(key, self.default_factory(key))


class EnumerateDict(dict):
    def __init__(self, start=0):
        self.next = start
        super().__init__()

    def __missing__(self, entry):
        new = self.next
        self[entry] = new
        self.next += 1
        return new

    def __len__(self):
        return self.next

# From: http://code.activestate.com/recipes/576694/
# XXX: Potential issue?: http://code.activestate.com/recipes/576694/#c9
class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        # sentinel node for doubly linked list
        end += [None, end, end]
        # key --> [key, prev, next]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

# TODO: Sanity testing...
