import nn_utils as nnu
import theano.tensor as T

# a0, b0 = 1., .5
a0, b0 = 6., 6.
eps_ind = 0.000001
sigma_ind = 1.
c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921


def sample_gauss(mu, std):
    return mu + std * nnu.srng.normal(mu.shape)


def sample_mgaus(mu, std_r, std_c):
    return mu + T.dot(std_r, nnu.srng.normal(mu.shape)).dot(std_c)


def sample_mgaus2(mu, std_r, std_c):
    return mu + T.dot(std_r.ravel().dimshuffle(0, 'x') * nnu.srng.normal(mu.shape), std_c)


def sample_mult_noise(sigma, shape):
    return 1 + sigma * nnu.srng.normal(shape)


def add_bias(x):
    return T.concatenate([x, T.ones((x.shape[0], 1))], axis=1)


def kldiv_gamma(a1, b1, a0=a0, b0=b0):
    return T.sum((a1 - a0)*nnu.Psi()(a1) - T.gammaln(a1) + T.gammaln(a0) + a0*(T.log(b1) - T.log(b0)) + a1*((b0 - b1)/b1))


class Layer(object):
    def __init__(self, params, nonlin='relu', priors=(0., 0., 0.), N=1):
        self.params = params
        self.N = N
        # self.amount_reg = amount_reg
        self.nonlin = nonlin
        self.nonlinearity = nnu.nonlinearities[nonlin]
        self.priors = priors

    def ff(self, x, sampling=True):
        out = self.f(x, sampling=sampling)
        act = self.nonlinearity(out)
        return act

    def set_params(self, params):
        for i, param in enumerate(self.params):
            param.set_value(params[i].get_value(borrow=False), borrow=False)

    def f(self, x, sampling=True):
        raise NotImplementedError()

    def get_reg(self):
        raise NotImplementedError()

    def get_priors(self):
        raise NotImplementedError()

    def __copy__(self):
        raise NotImplementedError()


from matrix_layers import layers_def
layers = layers_def