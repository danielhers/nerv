# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Assorted loss functions.

Version:    2013-02-06
Author:     Pontus Stenetorp    <pontus stenetorp se>
'''

from numpy import sum as numpy_sum
from numpy import log

def cross_entropy(predictions, output_data):
    return -numpy_sum(output_data * log(predictions))

def sum_squared(predictions, output_data):
    return (1 / 2) * numpy_sum((output_data - predictions) ** 2)
