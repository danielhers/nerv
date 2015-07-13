# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Neural network implementation, largely decoupled.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2014-02-06
'''

# TODO: Each class should probably be responsible for its own weight init.
# TODO: __slots__ for the vertices.
# TODO: Have we in a way re-invented the factory pattern? Ugh...

from collections import defaultdict
from copy import deepcopy
from math import fsum

from numpy import dot
from numpy import mean
from numpy import empty
from numpy import multiply
from numpy import product
from numpy import subtract
from numpy import transpose
from numpy import zeros
from scipy.linalg.blas import dger

from .init import init_layer
from .init import socher_2013_comp_mtrx
from .loss import cross_entropy
from .maths import softmax
from .maths import tanh
from .maths import tanh_prime
from .dag import DAG


# TODO: We might want to split this into separate classes.
#   Some have a special init, fan_in, etc.
class Vertex(object):
    def forward(self, net, model, loss=None):
        raise NotImplementedError

    def backward(self, net, model, gradient):
        raise NotImplementedError

    @classmethod
    def init(cls, weights):
        weights[:] = init_layer(weights.shape, weights.shape[0],
                fan_in=weights.shape[1])
        return weights.reshape(cls.weights_shape())

    @classmethod
    def weights_shape(cls):
        if cls.fan_out and cls.fan_in:
            return (cls.fan_out, cls.fan_in, )
        else:
            return (0, 0, )

    @classmethod
    def weights_size(cls):
        return product(cls.weights_shape())

    @classmethod
    def biases_shape(cls):
        if cls.fan_out and cls.fan_in:
            return (cls.fan_out, 1, )
        else:
            return (0, 0, )

    @classmethod
    def biases_size(cls):
        return product(cls.biases_shape())

    @classmethod
    def size(cls):
        return cls.weights_size() + cls.biases_size()


def static_source_vertex(fan_out_):
    class StaticSourceVertex(Vertex):
        fan_out = fan_out_
        fan_in = 0

        def __init__(self, activations):
            self.activations = activations

        def forward(self, net, model, loss=None):
            pass

        def backward(self, net, model, gradient):
            pass


    # XXX: Enormous hack, will fail if more than one kind is created...
    globals()[StaticSourceVertex.__name__] = StaticSourceVertex

    return StaticSourceVertex


# I have a nasty feeling that this use-case may be covered by the built-in...
class FallbackDict(dict):
    def __init__(self, fallback):
        super().__init__()
        self.fallback = fallback

    def __missing__(self, key):
        return self[self.fallback]


# XXX: HAS A BIAS TERM! BAAAD! WASTE!
# TODO: Does this structure require an ordered dictionary?
def keyed_source_vertex(dims, dic, missing_='<unk>', name_='keyed'):
    source_size = len(dic) * dims

    # Create a mapping between key and weight array region.
    assert missing_ in dic, "%s not present in dictionary" % missing_
    _slice_by_key = FallbackDict(missing_)
    for i, key in enumerate(dic):
        start = i * dims
        _slice_by_key[key] = slice(start, start + dims)


    class KeyedSourceVertex(Vertex):
        name = name_
        fan_out = dims
        fan_in = 1

        def __init__(self, key):
            super().__init__()
            self.key = key

        @classmethod
        def init(cls, weights):
            weights[:] = 0
            for key in dic:
                weights[cls.slice_by_key[key]] = dic[key]
            return weights

        @classmethod
        def weights_shape(cls):
            return (source_size, 1, )

        @classmethod
        def biases_shape(cls):
            return (0, 0, )

        slice_by_key = _slice_by_key

        def forward(self, net, model, loss=None):
            self.activations = model.weight[name_][self.slice_by_key[self.key]]

        def backward(self, net, model, gradient):
            _gradient = gradient.weight[name_][self.slice_by_key[self.key]]
            for child in net.children[self]:
                offset = 0
                for other_parent in net.parents[child]:
                    if other_parent is self:
                        break
                    offset += other_parent.activations.size
                _gradient += child.message[offset:offset + _gradient.size]


    # XXX: Enormous hack, will fail if more than one kind is created...
    # TODO: Could we set the Model name to something unique? namedtuple fails
    #   if there is more than one namedtuple with the same given name.
    globals()[KeyedSourceVertex.__name__] = KeyedSourceVertex

    return KeyedSourceVertex

def softmax_vertex(fan_out_, fan_in_, name_='softmax'):
    class SoftMaxVertex(Vertex):
        name = name_
        fan_out = fan_out_
        fan_in = fan_in_

        def __init__(self, target=None):
            super().__init__()
            self.target = target

        def forward(self, net, model, loss=None):
            parents = net.parents[self]
            size_sum = sum(parent.activations.size for parent in parents)
            assert fan_in_ == size_sum, "fan in mismatch: %d != %d" % (fan_in_, size_sum)
            offset = 0
            input_ = empty((fan_in_, 1))
            for parent in parents:
                size = parent.activations.size
                input_[offset:offset + size] = parent.activations
                offset += size

            activations = dot(model.weight[name_], input_)
            activations += model.bias[name_]
            softmax(activations, out=activations)

            if loss is not None and self.target is not None:
                loss[name_] += cross_entropy(activations, self.target)

            self.input = input_
            self.activations = activations

        def backward(self, net, model, gradient):
            if self.target is None:
                self.message[:] = 0
                return

            error = self.activations - self.target
            w_gradient = multiply(error, self.input.T)

            gradient.weight[name_] += w_gradient
            gradient.bias[name_] += error

            self.message = dot(model.weight[name_].T, error)


    # XXX: Enormous hack, will fail if more than one kind is created...
    globals()[SoftMaxVertex.__name__] = SoftMaxVertex

    return SoftMaxVertex

# TODO: Could be handed activation function, etc.?
def rnn_vertex(dim, num_inputs_, name_='rnn'):
    fan_out_ = dim
    fan_in_ = dim * num_inputs_

    class RNNVertex(Vertex):
        name = name_
        fan_out = fan_out_
        fan_in = fan_in_
        num_inputs = num_inputs_

        @classmethod
        def init(cls, weights):
            weights = weights.reshape(cls.weights_shape())
            weights[:] = socher_2013_comp_mtrx(cls.fan_out, cls.num_inputs)
            return weights

        def forward(self, net, model, loss=None):
            parents = net.parents[self]
            size_sum = sum(parent.activations.size for parent in parents)
            assert fan_in_ == size_sum, "fan in mismatch: %d != %d" % (fan_in_, size_sum)
            input_ = empty((fan_in_, 1))
            offset = 0
            for parent in parents:
                size = parent.activations.size
                input_[offset:offset + size] = parent.activations
                offset += size

            # Calculate the activations.
            activations = dot(model.weight[name_], input_)
            activations += model.bias[name_]
            tanh(activations, out=activations)

            self.input = input_
            self.activations = activations

        def backward(self, net, model, gradient):
            children = net.children[self]
            if not children:
                self.message[:] = 0
                return

            # Collect the incoming message from all children.
            incoming_message = zeros((fan_out_, 1))
            for child in children:
                offset = 0
                for other_parent in net.parents[child]:
                    if other_parent is self:
                        break
                    offset += other_parent.fan_out
                incoming_message += child.message[offset:
                        offset + incoming_message.size]

            # Calculate the gradients.
            back = tanh_prime(self.activations) * incoming_message
            dger(1.0, self.input.T, back.T, a=gradient.weight[name_].T,
                    overwrite_a=True)
            gradient.bias[name_] += back

            self.message = dot(transpose(model.weight[name_]), back)


    # XXX: Enormous hack, will fail if more than one kind is created...
    globals()[RNNVertex.__name__] = RNNVertex

    return RNNVertex

# TODO: Could be handed activation function, etc.?
def average_vertex(dim, name_='average'):
    fan_out_ = dim
    fan_in_ = dim

    class AverageVertex(Vertex):
        name = name_
        fan_out = fan_out_
        fan_in = fan_in_

        @classmethod
        def init(cls, weights):
            weights = weights.reshape(cls.weights_shape())
            weights[:] = socher_2013_comp_mtrx(cls.fan_out, 1)
            return weights

        def forward(self, net, model, loss=None):
            parents = net.parents[self]
            input_ = empty((fan_in_, len(parents)))
            for i, parent in enumerate(parents):
                input_[:, i] = parent.activations.squeeze()

            # Calculate the activations.
            activations = dot(model.weight[name_], input_)
            activations = mean(activations, axis=1).reshape(-1, 1)
            activations += model.bias[name_]
            tanh(activations, out=activations)

            self.input = input_
            self.activations = activations

        def backward(self, net, model, gradient):
            children = net.children[self]
            if not children:
                self.message[:] = 0
                return

            # Collect the incoming message from all children.
            incoming_message = zeros((fan_out_, 1))
            for child in children:
                incoming_message += child.message

            # Calculate the gradients.
            back = tanh_prime(self.activations) * incoming_message
            dger(1.0, self.input.T, back.T, a=gradient.weight[name_].T,
                 overwrite_a=True)
            gradient.bias[name_] += back

            self.message = dot(transpose(model.weight[name_]), back)


    # XXX: Enormous hack, will fail if more than one kind is created...
    globals()[AverageVertex.__name__] = AverageVertex

    return AverageVertex


def net_model(_vertice_classes):
    # TODO: Impl. __getstate__ and __setstate__ to side-step "init-keys" ref.
    class Model(object):
        vertice_classes = _vertice_classes

        def __init__(self):
            # Use a single slice of contiguous memory for the parameters
            #   (weights and biases).
            self.params = empty((sum(c.size() for c in _vertice_classes), 1, ))

            # Assign a view for the parameters of each of the classes.
            self.weight = {}
            self.bias = {}

            self._init_keys()

        def _init_keys(self):
            params = self.params
            offset = 0
            for v_class in (c for c in _vertice_classes if c.size()):
                key = v_class.name

                # Assign a portion of the parameters to the weights.
                w_size = v_class.weights_size()
                w_slice = params[offset:offset + w_size]
                self.weight[key] = v_class.init(w_slice)
                offset += w_size

                # Assign a portion of the parameters to the biases.
                b_size = v_class.biases_size()
                biases = params[offset:offset + b_size].reshape(
                        v_class.biases_shape())
                biases[:] = 0
                self.bias[key] = biases
                offset += b_size

        def clear(self):
            self.params[:] = 0

        # XXX: Name is a bit confusing...
        def gradient(self):
            gradient = deepcopy(self)
            # Note: After the copy the weight/bias views will be invalid,
            #   re-initialise them.
            gradient._init_keys()
            gradient.clear()
            return gradient

        def loss(self, nets, loss=None, normalise=True):
            if loss is None:
                loss = Loss()

            for net in nets:
                net.forward(self, loss)

            if normalise:
                loss.normalise(len(nets))

            return loss

        # XXX: Re-consider the set-up.
        def loss_and_gradient(self, nets, loss=None, gradient=None,
                no_loss=False, normalise=True):
            # TODO: Use the loss method?
            if loss is None and not no_loss:
                loss = Loss()
            if gradient is None:
                gradient = self.gradient()

            for net in nets:
                net.forward(self, loss=loss)
                net.backward(self, gradient=gradient)

            if normalise:
                if loss is not None:
                    loss.normalise(len(nets))
                gradient.params /= len(nets)

            return (loss, gradient, )

        # XXX: Yet another hideous hack... This to share the vocabulary slice
        #   mapping.
        def __getstate__(self):
            dic = self.__dict__.copy()
            vc_dic = {}

            for vc in self.vertice_classes:
                try:
                    vc_dic[vc] = vc.slice_by_key
                except AttributeError:
                    pass

            dic['vc_dic'] = vc_dic
            return dic

        # XXX: Yet another hideous hack... This to share the vocabulary slice
        #   mapping.
        def __setstate__(self, state):
            vc_dic = state['vc_dic']
            del state['vc_dic']

            for vc in self.vertice_classes:
                try:
                    vc.slice_by_key
                except AttributeError:
                    pass
                else:
                    vc.slice_by_key = vc_dic[vc]

            self.__dict__ = state

        # TODO: Default to single or multiple nets?
        def forward(self, net, loss=None):
            net.forward(self, loss=loss)

        # TODO: Default to single or multiple nets?
        def backward(self, net, gradient=None):
            return net.backward(self, gradient=gradient)


    # XXX: Enormous hack, will fail if more than one kind is created...
    globals()[Model.__name__] = Model

    return Model


class Net(DAG):
    def forward(self, model, loss=None):
        for node in self.topological_sort():
            node.forward(self, model, loss=loss)

    def backward(self, model, gradient=None):
        if gradient is None:
            gradient = model.gradient()

        for node in self.topological_sort(reverse=True):
            node.backward(self, model, gradient)

        return gradient


class Loss(dict):
    def __missing__(self, key):
        return 0.0

    def normalise(self, n):
        for k in self:
            self[k] /= n

    def total(self):
        return fsum(self.values())
