import sys
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict


class BaseOptimizer(object):
    """Base optimizer class
    """
    def __init__(self, objectives, params, gradients=None, batch_size=100, polyak=True, beta3=0.999):
        self.batch_size = batch_size
        self.batches_test = 1
        self.polyak = polyak
        self.beta3 = beta3
        if gradients is None:
            self.gradients = T.grad(T.sum(objectives), params, disconnected_inputs='warn', add_names=True)
        else:
            self.gradients = gradients

    def train(self, data, verbose=False):
        batches = np.arange(0, data[0].shape[0], self.batch_size)
        batches = np.append(batches, data[0].shape[0])  # add the remaining datapoints
        lb = 0
        rounding = lambda x: ['%.3f' % i for i in x]
        for j in xrange(len(batches) - 1):
            inp = [d[batches[j]:batches[j + 1]] for d in data]
            objectives = np.array(self._ascent(*inp))
            self._update_inf()
            if np.isnan(objectives).any():
                raise Exception('NaN objective!')
            lb += objectives / (len(batches) - 1.)
            if verbose:
                sys.stdout.write("\rBatch:{0}, Objectives:{1}, Total:{2}".format(str(j + 1) +
                                                                                 '/' + str(len(batches) - 1),
                                                                                 str(rounding((objectives).tolist())),
                                                                                 str(rounding(lb.tolist()))))
                sys.stdout.flush()
        if verbose:
            print

        return lb

    def _ascent(self):
        raise NotImplementedError()

    def _eval(self):
        raise NotImplementedError()

    def _update_inf(self):
        raise NotImplementedError()

    def get_updates_eval(self, params_inf, params):
        """Keep an exponential moving average of the parameters, which will be used for evaluation
        """
        updates_eval = OrderedDict()

        itinf = theano.shared(0., name='itinf')
        updates_eval[itinf] = itinf + 1.
        fix3 = 1. - self.beta3**(itinf + 1.)

        for i in xrange(len(params)):
            if self.polyak:
                if 'scalar' in params_inf[i].name:
                    avg = theano.shared(np.cast[theano.config.floatX](0.), name=params_inf[i].name + '_avg')
                else:
                    avg = theano.shared(params_inf[i].get_value() * 0., name=params_inf[i].name + '_avg',
                                        broadcastable=params_inf[i].broadcastable)

                avg_new = self.beta3 * avg + (1. - self.beta3) * params[i]
                updates_eval[avg] = T.cast(avg_new, theano.config.floatX)
                updates_eval[params_inf[i]] = T.cast(avg_new / fix3, theano.config.floatX)
            else:
                updates_eval[params_inf[i]] = params[i]

        return updates_eval


class Adam(BaseOptimizer):
    """Adam optimizer for an objective function
    """
    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None,
                 alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=100, polyak=True, beta3=0.999,
                 **kwargs):

        super(Adam, self).__init__(objectives, params, gradients=gradients, batch_size=batch_size, polyak=polyak,
                                   beta3=beta3)

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        try:
            self.max_drop = kwargs.pop('max_drop')
        except:
            self.max_drop = 1.
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)
        inputs = inputs

        # evaluate all the objectives and update parameters
        self._ascent = theano.function(inputs, objectives, updates=updates, on_unused_input='warn', mode='FAST_RUN')
        # evaluate all the objectives and (optionally) use a moving average for the parameters
        self._update_inf = theano.function([], [], updates=updates_eval, on_unused_input='warn', mode='FAST_RUN')
        self._eval = theano.function(inputs, objectives_eval, on_unused_input='ignore', mode='FAST_RUN')
        print 'Adam', 'alpha:', alpha, 'beta1:', beta1, 'beta2:', beta2, 'epsilon:', self.epsilon, \
            'batch_size:', self.batch_size, 'polyak:', polyak, 'beta3:', beta3

    def get_updates(self, params, grads):
        updates = OrderedDict()

        it = theano.shared(0., name='it')
        updates[it] = it + 1.

        fix1 = 1. - self.beta1**(it + 1.)  # To make estimates unbiased
        fix2 = 1. - self.beta2**(it + 1.)  # To make estimates unbiased

        for i in xrange(len(grads)):
            gi = grads[i]

            # mean_squared_grad := E[g^2]_{t-1}
            if 'scalar' in params[i].name:
                mom1 = theano.shared(np.cast[theano.config.floatX](0.))
                mom2 = theano.shared(np.cast[theano.config.floatX](0.))
            else:
                mom1 = theano.shared(params[i].get_value() * 0., broadcastable=params[i].broadcastable)
                mom2 = theano.shared(params[i].get_value() * 0., broadcastable=params[i].broadcastable)

            # Update moments
            mom1_new = self.beta1 * mom1 + (1. - self.beta1) * gi
            mom2_new = self.beta2 * mom2 + (1. - self.beta2) * T.sqr(gi)

            # Compute the effective gradient
            corr_mom1 = mom1_new / fix1
            corr_mom2 = mom2_new / fix2
            effgrad = corr_mom1 / (T.sqrt(corr_mom2) + self.epsilon)

            # Do update
            w_new = params[i] + self.alpha * effgrad

            if 'dropout_alpha' in params[i].name:
                maxd = np.log(np.sqrt(self.max_drop))
                w_new = T.clip(w_new, -4., maxd)

            # Apply update
            updates[params[i]] = T.cast(w_new, theano.config.floatX)
            updates[mom1] = T.cast(mom1_new, theano.config.floatX)
            updates[mom2] = T.cast(mom2_new, theano.config.floatX)
        return updates


optimizers = {'adam': Adam}
