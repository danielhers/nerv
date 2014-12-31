# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
A Directed Multigraph with some caching of attributes for better run-time
performance.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-12
'''

# TODO: Implement remove_edge?
# TODO: Could use more clever caching rather than full invalidation.
# TODO: The above cache would also allow us to avoid a method call.
# TODO: Consider an incremental topological sort?
# TODO: Sanity checking?

from collections import defaultdict

from lib.structs import OrderedSet
from lib.structs import enum

VertexType = enum('SOURCE', 'SINK', 'INTERNAL')


class DAG(object):
    def __init__(self):
        self.parents = defaultdict(OrderedSet)
        self.children = defaultdict(OrderedSet)
        self.vertices = set()
        # Cache(s).
        self._topo_sort = None

    def __iter__(self):
        return iter(self.vertices)

    def add_edge(self, parent, child):
        self.vertices.add(parent)
        self.vertices.add(child)
        self.parents[child].add(parent)
        self.children[parent].add(child)

        # Invalidate cache(s).
        self._topo_sort = None

    def typed_it(self):
        for vertex in self:
            if not self.parents[vertex]:
                yield (VertexType.SOURCE, vertex)
            elif not self.children[vertex]:
                yield (VertexType.SINK, vertex)
            else:
                yield (VertexType.INTERNAL, vertex)

    def sources(self):
        return (v for t, v in self.typed_it() if t == VertexType.SOURCE)

    def sinks(self):
        return (v for t, v in self.typed_it() if t == VertexType.SINK)

    def internals(self):
        return (v for t, v in self.typed_it() if t == VertexType.INTERNAL)

    # http://en.wikipedia.org/wiki/Topological_sorting
    def _topological_sort(self):
        tsort = []
        processed = set()

        # Start a Depth-First Search (DFS) from each vertex and add each
        #   vertex after the search has added their children to the order.
        #   Lastly, reverse the order to gain the topological sort.
        for start in self:
            # No need to process a vertex twice.
            if start in processed:
                continue

            # Start the DFS from this vertex.
            queue = [start]
            while queue:
                current = queue[-1]
                # If we have already processed this vertex it means that all
                #   of its descendants have been processed and we can safely
                #   add it to the topological sort.
                if current in processed:
                    tsort.append(current)
                    queue.pop()
                    continue

                # Process the vertex by adding each of its unprocessed
                #   children to the queue.
                for child in self.children[current]:
                    if child not in processed:
                        queue.append(child)
                processed.add(current)

        return reversed(tsort)

    def topological_sort(self, reverse=False):
        if self._topo_sort is not None:
            topo_sort = self._topo_sort
        else:
            topo_sort = tuple(self._topological_sort())
            self._topo_sort = topo_sort

        return topo_sort if not reverse else reversed(topo_sort)
