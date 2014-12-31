**nerv**, modelling composition for Directed Acyclic Graphs in Python.

**Warning:** This is largely **unsupported** software, no extensions are
planned and development has stopped. It is however fully functional and has
served as a test-bed for compositional models between 2012 and 2014. Designing
a Python library like this that is performant, easy to debug and elegant
is difficult. The main author has now switched to using [Julia][julia]
for most of his research, but it is not entirely unlikely that he may take
another stab at a library similar to nerve, if time permits it, over the next
few years.

[julia]: http://julialang.org/

# Installation #

## Required ##

On a reasonably modern Debian-based system, use the following to install all
of the required dependencies:

    sudo apt-get install python3 python3-numpy python3-scipy

To make sure that everything is hunky-dory, run the sanity checks:

    make sanity

Then proceed to using the library, happy hacking!

## Optional ##

### BLAS ###

Most likely you want to use OpenBLAS as your BLAS back-end, it is fast (albeit
maybe not always the most reliable):

    sudo apt-get libopenblas-base

### Cython ###

You can improve the speed of the library using Cython, but it is not required:

    sudo apt-get cython3 libopenblas-dev
    make

### line\_profiler ###

If you want line-by-line performance analysis you will need the `line_profiler`
module, while the official version does not support Python 3, there is a
patched version available.

    sudo apt-get install python3-pip
    sudo pip3 install \
        https://bitbucket.org/kmike/line_profiler/get/default.tar.gz

# Mission Statement #

Deep Learning, in particular feature learning, allows for an amasing new set
of models for composition, especially for Natural Language Processing tasks.
However, a vast majority of the community is used to the standard feature
engineering, then throwing things into a Support Vector Machine or linear
classifier development cycle.

We want to change this, while it may never become as easy as generating a
simple sparse matrix that is fed into an external linear classifier, we will
do our utmost to bring the rest of the community on-board without having to
read highly optimised research code or deriving methods from papers.

Simply put, we aim for:

1. Easy of use.
2. General applicability.
3. Performance.

In that order.

# Development Philosophy #

1. Implementation.
2. Sanity checks.
3. Performance tests.
4. Is a function or region too slow? Optimise, in this order:
    1. Pure Python.
    2. Cython.
5. GOTO 1

# Citing #

If you find this library useful in some way and want to provide academic
credit, the best way to do so is probably to cite

    @InProceedings{stenetorp2013transition,
        author      = {Stenetorp, Pontus},
        title       = {Transition-based Dependency Parsing
            Using Recursive Neural Networks},
        booktitle   = {Deep Learning Workshop at the 2013 Conference on
            Neural Information Processing Systems (NIPS)},
        address     = {Lake Tahoe, Nevada, USA},
        year        = {2013},
        month       = {December},
    }

since this library served as the basis for the model presented in the paper.
