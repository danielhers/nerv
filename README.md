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

# Usage #

```python
    # We will use a Stanford Sentiment Treebank-like example where we have the
    #   phrase "This burger isn't bad" and annotate each sub-phrase of
    #   the sentence with a sentiment label.
    tokens = set(('This', 'burger', 'is', "n't", 'bad'))

    # The dimensionality of our token representations.
    dims = 32

    from collections import OrderedDict
    from nerv.init import random_uniform

    # Mappings between each token and a randomly initialised representation.
    reprs = OrderedDict([(t, random_uniform(dims)) for t in tokens])

    # There are two core concepts, nets and models. Nets are Directed Acyclic
    #   Graphs (DAGs) that consists of vertices of varying types. These can be
    #   token representations, compositional, predictive, etc. A model is
    #   simply a collection of weights that can be applied to a net.

    from numpy import array

    # One-hot sentiment labels as column vectors.
    very_neg = array((1, 0, 0, 0, 0)).reshape(-1, 1)
    neg      = array((0, 1, 0, 0, 0)).reshape(-1, 1)
    neutral  = array((0, 0, 1, 0, 0)).reshape(-1, 1)
    pos      = array((0, 0, 0, 1, 0)).reshape(-1, 1)
    very_pos = array((0, 0, 0, 0, 1)).reshape(-1, 1)
    labels = (very_neg, neg, neutral, pos, very_pos)

    from nerv.net import keyed_source_vertex
    from nerv.net import rnn_vertex
    from nerv.net import softmax_vertex

    # Generate classes for each desired vertex.
    TokenVertex = keyed_source_vertex(dims, reprs)
    # Composes two vertices of a desired dimensionality.
    CompVertex = rnn_vertex(dims, 2)
    # Predict one out of several sentiment labels.
    SentVertex = softmax_vertex(len(labels), dims)

    from nerv.net import net_model

    # Generate a model class that is applicable to our vertices.
    Model = net_model((TokenVertex, CompVertex, SentVertex))

    from nerv.net import Net

    # Let us now construct a net for our example sentence.
    net = Net()

    # The sources in our DAG, the tokens.
    this   = TokenVertex('This')
    burger = TokenVertex('burger')
    is_    = TokenVertex('is')
    nt     = TokenVertex("n't")
    bad    = TokenVertex('bad')

    # Add a vertex predicting the sentiment for each token.
    net.add_edge(this  , SentVertex(neutral))
    net.add_edge(burger, SentVertex(neutral))
    net.add_edge(is_   , SentVertex(neutral))
    net.add_edge(nt    , SentVertex(neutral))
    net.add_edge(bad   , SentVertex(very_neg))

    # The first level of composition.
    this_burger = CompVertex()
    is_nt       = CompVertex()
    net.add_edge(this  , this_burger)
    net.add_edge(burger, this_burger)
    net.add_edge(is_   , is_nt)
    net.add_edge(nt    , is_nt)

    # Attach a vertex predicting the sentiment for each composed vertex.
    net.add_edge(this_burger, SentVertex(neutral))
    net.add_edge(is_nt      , SentVertex(neg))

    # Rince and repeat.
    is_nt_bad             = CompVertex()
    this_burger_is_nt_bad = CompVertex()
    net.add_edge(is_nt      , is_nt_bad)
    net.add_edge(bad        , is_nt_bad)
    net.add_edge(this_burger, this_burger_is_nt_bad)
    net.add_edge(is_nt_bad  , this_burger_is_nt_bad)
    net.add_edge(is_nt_bad  , SentVertex(pos))
    sent_sentiment = SentVertex(pos)
    net.add_edge(this_burger_is_nt_bad, sent_sentiment)

    # We can now initialise a model and apply it to the net.
    model = Model()
    model.forward(net)

    # Each SentVertex will now carry a prediction, for example:
    sent_sentiment.activations

    # Returns the current loss and gradient for a given set of parameters.
    def f(params):
        # It is not necessary to make an explicit update here if you send
        #   the model parameters to the minimisation function since it will
        #   do this internally at each step.
        #model.params[:] = params
        # Any mini-batch logic would go here.
        (loss, grad) = model.loss_and_gradient((net, ))
        # Add any desired regularisation here.
        return (loss, grad.params)

    from nerv.optimise import fmin_adagrad

    iteration = 0
    for _, loss, _ in fmin_adagrad(f, model.params):
        iteration += 1
        # Uncomment to inspect the loss, it should go down rapidly.
        #print(loss.total())
        if iteration >= 42:
            break
    # At this point you have a, somewhat, trained model.
```

# Installation #

## Required ##

On a reasonably modern Debian-based system, use the following to install all
of the required dependencies:

    sudo apt-get install python3 python3-numpy python3-scipy

To make sure that everything is hunky-dory, run the sanity checks:

    make sanity

Then proceed to using the library, happy hacking!

## Optional ##

### Performance Suite ###

There is an optional performance suite included, to run it type

    make perf

and you should see numbers around a few hundred nets per second, depending
on your hardware.

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
