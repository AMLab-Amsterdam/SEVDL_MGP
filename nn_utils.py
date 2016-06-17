import theano
import theano.tensor as T
import numpy as np
import scipy.special as sp
from theano.scalar.basic import complex_types, discrete_types
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=12345)
prng = np.random.RandomState(12345)
sigma_init, eps = 0.01, 1e-8


def change_random_seed(seed):
    global prng, srng
    srng = RandomStreams(seed=seed)
    prng = np.random.RandomState(seed)


def randmat2(dim1, dim2, name, mu=0., sigma=None, type_dist='normal'):
    if not sigma:
        sigma = sigma_init
    if type_dist == 'normal':
        val = prng.normal(mu, sigma, (dim1, dim2))
    else:
        boundary1, boundary2 = sigma, sigma
        val = prng.uniform(-boundary1, boundary2, (dim1, dim2)) + mu
    return theano.shared(value=val.astype(theano.config.floatX), name=name, borrow=False)


def randmat(dim1, dim2, name, type_init='xavier', type_dist='normal'):
    if 'sigma' in name:
        val = prng.normal(0, sigma_init, (dim1, dim2))
        return theano.shared(value=val.astype(theano.config.floatX), name=name, borrow=False)
    else:
        if type_init == 'xavier':
            bound = np.sqrt(1. / (dim1 - 1))
        elif type_init == 'xavier2':
            bound = np.sqrt(2. / ((dim1 - 1) + dim2))
        elif type_init == 'he':
            bound = np.sqrt(2. / (dim1 - 1))
        elif type_init == 'he2':
            bound = np.sqrt(4. / ((dim1 - 1) + dim2))
        elif type_init == 'regular':
            bound = sigma_init
        else:
            raise Exception()
        if type_dist == 'normal':
            val1 = prng.normal(0., bound, (dim1 - 1, dim2))  # actual weight initialization
        else:
            val1 = prng.uniform(-bound, bound, (dim1 - 1, dim2))
        val2 = np.zeros((1, dim2))
        val = np.concatenate([val1, val2], axis=0)
        return theano.shared(value=val.astype(theano.config.floatX), name=name, borrow=False)


def multvector(dim, mult, name):
    return theano.shared(value=(mult * np.ones((dim,))).astype(theano.config.floatX), name=name, borrow=False)


def randvector(dim, name, mu=0., sigma=sigma_init):
    if 'sigma_row_mgauss' in name:
        val1 = prng.normal(mu, sigma, (dim-1,)).astype(theano.config.floatX)
        val2 = prng.normal(-4, sigma, (1,)).astype(theano.config.floatX)
        val = np.concatenate([val1, val2], axis=0)
    else:
        val = prng.normal(mu, sigma, (dim,)).astype(theano.config.floatX)
    return theano.shared(value=val, name=name, borrow=False)


def tscalar(val, name):
    return theano.shared(np.cast[theano.config.floatX](val), name)


'''
Nonlinear functions
'''
relu = T.nnet.relu
elu = lambda x, a=1.: T.switch(x < 0, a*(T.exp(x) - 1.), x)
linear = lambda x: x
nonlinearities = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softmax': T.nnet.softmax, 'softplus': T.nnet.softplus,
                  'relu': relu, 'linear': linear, 'elu': elu}


def log_f(string, f='log.txt'):
    with open('logs/' + f, 'ab') as handle:
        handle.write(string + '\n')
        print string


class Polygamma(theano.Op):
    """
    This creates an Op that produces the polygamma function
    """
    __props__ = ("n")

    def __init__(self, n):
        self.n = n
        super(Polygamma, self).__init__()

    def make_node(self, x):
        # check that the theano version has support for __props__.
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = sp.polygamma(self.n, x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x, = inputs
        gz, = output_grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]
        return [gz * Polygamma(self.n + 1)(x)]


class Psi(theano.Op):
    """
    This creates an Op that produces the digamma function
    """

    def __init__(self):
        super(Psi, self).__init__()

    def make_node(self, x):
        # check that the theano version has support for __props__.
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = sp.psi(x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x, = inputs
        gz, = output_grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]
        return [gz * Polygamma(1)(x)]
