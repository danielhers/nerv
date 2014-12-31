# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Functionality primarily related to natural languages.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-03-12
'''

from collections import defaultdict
from itertools import chain
from itertools import islice
from itertools import permutations
from math import ceil
from random import gammavariate
from string import ascii_lowercase

from numpy.random import zipf

def _strseq():
    return (''.join(t) for t in chain.from_iterable(
        permutations(ascii_lowercase, r=n)
        for n in range(len(ascii_lowercase))))

# Naive sequence of unique tokens according to the Zipfian distribution, does
#   not yield realistic sequences (you need a language model for that) but
#   does yield a somewhat realistic distribution over the unique tokens that
#   appear.
def zipfgen():
    tok_it = _strseq()
    vocab = defaultdict(lambda : next(tok_it))
    while True:
        yield vocab[zipf(2.0)]

# Random sentence lengths using a Gamma distribution along the lines of
#   Sigurd et al. (2004). The parameters were estimated using the
#   2013-12-03 English Wikipedia dump, tokenised and sentence split using the
#   CoreNLP tools.
def randslen():
    # Ceil ensures that we don't end up with zero length.
    return int(ceil(gammavariate(3.08095294271, 8.01572810369) + 0.0))

def sentgen(min_len=1):
    tok_it = zipfgen()
    while True:
        randlen = randslen()
        if randlen < min_len:
            continue
        yield islice(tok_it, randlen)
