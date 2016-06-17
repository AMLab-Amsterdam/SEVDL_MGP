import theano
import theano.tensor as T
import nn_utils as nnu
import numpy as np
import time
import optimizer as opt
import layers as LL
from copy import copy
from collections import OrderedDict
from layers import add_bias, kldiv_gamma, a0, b0


class VMGNet(object):
    def __init__(self, N, dimx, dimy, dimh=(100, 100), nonlinearity='relu', learning_rate=0.001, n_iter=100,
                 batch_size=100, priors=(0., 0., 0.), logtxt='VMGNet.txt', optimizer='adam', polyak=True, beta3=0.999,
                 seed=1234, task_type='classification', sampling_pred=False, type_init='he2', n_inducing=50,
                 ind_noise_lvl=0.1, **kwargs):
        # network topology
        self.dimx = dimx
        self.dimy = dimy
        self.dimh = dimh
        self.nonlinearity = nonlinearity

        self.n_inducing = n_inducing
        self.ind_noise_lvl = ind_noise_lvl

        self.N = N
        self.N_valid = N
        if 'n_valid' in kwargs:
            self.N_valid = kwargs.pop('n_valid')

        self.priors = priors
        self.task_type = task_type
        if self.task_type not in ['regression', 'classification']:
            raise Exception()
        # optimization parameters
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.sampling_pred = sampling_pred

        self.alg_opt = opt.optimizers[optimizer]
        self.polyak = polyak
        self.beta3 = beta3
        self.layer = LL.layers['mgdl_ff'] if self.task_type == 'classification' else LL.layers['mgdl_lp']
        self.type_init = type_init

        # misc parameters
        self.logtxt = logtxt
        self.seed = seed
        nnu.change_random_seed(seed)

        nnu.log_f('Initialized VMGNet with dimx: ' + str(dimx) + ', dimy: ' + str(dimy) + ', dimh: ' + str(dimh) +
                  ', N: ' + str(N) + ', N_valid: ' + str(self.N_valid) + ', nonlinearity: ' + nonlinearity +
                  ', learning_rate: ' + str(learning_rate) + ', seed: ' + str(seed) + ', iterations: ' + str(n_iter) +
                  ', batch_size: ' + str(batch_size) + ', priors: ' + str(priors) + ', layer_type: ' + str(self.layer) +
                  ', task_type: ' + task_type + ', sampling_pred: ' + str(sampling_pred) + ', type_init: ' + type_init +
                  ', n_pseudo: ' + str(n_inducing) + ', upper_bound_noise: ' + str(ind_noise_lvl), f=self.logtxt)

    def _create_parameters(self):
        """
        Instantiate the parameters of the network
        :return:
        """
        self.extra, self.extra_inf = OrderedDict(), OrderedDict()

        # input layer
        gin = self.layer(self.dimx + 1, self.dimh[0], 'in', priors=self.priors, N=self.N, nonlin=self.nonlinearity,
                         type_init=self.type_init, n_inducing=self.n_inducing, noise_lvl=self.ind_noise_lvl)
        layers = [gin]
        # remaining hidden layers
        self.hidden_params = []
        for i, h in enumerate(self.dimh[1:]):
            gh = self.layer(self.dimh[i] + 1, h, 'h' + str(i + 1), priors=self.priors, N=self.N, nonlin=self.nonlinearity,
                            type_init=self.type_init, n_inducing=self.n_inducing, noise_lvl=self.ind_noise_lvl)
            layers.append(gh)
        gout = self.layer(self.dimh[-1] + 1, self.dimy, 'out', priors=self.priors, nonlin='linear',
                          type_init=self.type_init, N=self.N, n_inducing=self.n_inducing, noise_lvl=self.ind_noise_lvl)
        layers.append(gout)

        if self.task_type == 'regression':
            a1, b1 = nnu.multvector(self.dimy, np.log(a0), 'out_a1'), nnu.multvector(self.dimy, np.log(b0), 'out_b1')
            a1inf, b1inf = nnu.multvector(self.dimy, np.log(a0), 'out_a1_inf'), nnu.multvector(self.dimy, np.log(b0),
                                                                                               'out_b1_inf')
            self.extra['a1'] = a1; self.extra['b1'] = b1
            self.extra_inf['a1'] = a1inf; self.extra_inf['b1'] = b1inf

        self.layers = layers
        self.layers_inf = [copy(layer) for layer in layers]
        for layeri in self.layers_inf:
            layeri.N = self.N_valid

    def gaussian_like(self, y, mu, **kwargs):
        a1, b1 = kwargs.pop('a1'), kwargs.pop('b1')
        etau, elogtau = a1 / b1, nnu.Psi()(a1) - T.log(b1)
        return T.sum(.5 * elogtau - .5 * T.log(2*np.pi) - (.5 * etau * T.sqr(y - mu)), axis=1)

    def _training(self):
        """
        Define the computational graph
        :return:
        """
        self.x = T.matrix('x')
        self.y_ = T.ivector('y') if self.task_type == 'classification' else T.matrix('y')

        # first estimate the regularization terms
        reg = self.layers[0].get_reg()
        regi = self.layers_inf[0].get_reg()
        for i in xrange(len(self.layers[1:])):
            regs = self.layers[i + 1].get_reg()
            regsi = self.layers_inf[i + 1].get_reg()
            for k in xrange(len(reg)):
                reg[k] += regs[k]
                regi[k] += regsi[k]

        # now estimate the likelihood term
        h, hinf = [self.x], [self.x]
        for i in xrange(len(self.dimh)):
            dot = self.layers[i].ff(add_bias(h[-1]))
            dot_inf = self.layers_inf[i].ff(add_bias(hinf[-1]))
            h.append(dot)
            hinf.append(dot_inf)

        # output
        lin_dot = self.layers[len(self.dimh)].ff(add_bias(h[-1]))
        lin_dot_inf = self.layers_inf[len(self.dimh)].ff(add_bias(hinf[-1]))

        # error
        if self.task_type == 'classification':
            y, yinf = T.nnet.softmax(lin_dot), T.nnet.softmax(lin_dot_inf)
            err = -T.nnet.categorical_crossentropy(y, self.y_)
            erri = -T.nnet.categorical_crossentropy(yinf, self.y_)
        elif self.task_type == 'regression':
            out_y, out_yi = self.revy(lin_dot), self.revy(lin_dot_inf)
            a1, a1i = T.exp(self.extra['a1']), T.exp(self.extra_inf['a1'])
            b1, b1i = T.exp(self.extra['b1']), T.exp(self.extra_inf['b1'])
            err = self.gaussian_like(self.y_, out_y, a1=a1, b1=b1)
            erri = self.gaussian_like(self.y_, out_yi, a1=a1i, b1=b1i)

        loss_obj = T.mean(err)#.sum() / T.cast((1. * self.x.shape[0]), theano.config.floatX)
        loss_obj_inf = T.mean(erri)#.sum() / T.cast((1. * self.x.shape[0]), theano.config.floatX)
        objectives = [loss_obj] + reg
        objectives_inference = [loss_obj_inf] + regi

        return [self.x, self.y_], [objectives, objectives_inference]

    def _inference(self, x):
        # MAP and sample predictions
        # input layer
        h, hs = [x], [x]
        # hidden layers
        for i in xrange(len(self.dimh)):
            out = self.layers_inf[i].ff(add_bias(h[-1]), sampling=False)
            outs = self.layers_inf[i].ff(add_bias(hs[-1]), sampling=True)
            h.append(out)
            hs.append(outs)

        # output layer
        hout = self.layers_inf[len(self.dimh)].ff(add_bias(h[-1]), sampling=False)
        houts = self.layers_inf[len(self.dimh)].ff(add_bias(hs[-1]), sampling=True)
        if self.task_type == 'classification':
            ypred, ypreds = T.nnet.softmax(hout), T.nnet.softmax(houts)
            self.predict_mean = theano.function([x], T.argmax(ypred, axis=1))
            self.predict_sample = theano.function([x], T.argmax(ypreds, axis=1))
        else:
            self.predict_mean = theano.function([x], self.revy(hout))
            self.predict_sample = theano.function([x], self.revy(houts))

    def predict(self, x, samples=1, batch_size_p=100):
        y_ = np.zeros((samples, x.shape[0], self.dimy))
        chunks = [range(i, i+batch_size_p) for i in xrange(0, x.shape[0], batch_size_p)]
        chunk_ = [elem for elem in chunks[-1] if elem < x.shape[0]]  # remove indices that exceed range
        chunks[-1] = chunk_

        if samples == 1:
            # MAP estimation
            for lc, chunk in enumerate(chunks):
                if self.task_type == 'classification':
                    pred = self.predict_mean(x[chunk, :].astype(np.float32))
                    y_[0, chunk, pred.astype(np.int32)] = 1
                else:
                    y_[0, chunk] = self.predict_mean(x[chunk, :].astype(np.float32))
            if self.task_type == 'classification':
                return np.argmax(y_[0], axis=1)
            return y_[0]

        for ksample in xrange(samples):
            for chunk in chunks:
                if self.task_type == 'classification':
                    pred = self.predict_sample(x[chunk, :].astype(np.float32))
                    y_[ksample, chunk, pred.astype(np.int32)] = 1
                else:
                    y_[ksample, chunk] = self.predict_sample(x[chunk, :].astype(np.float32))
        avg_y = y_.mean(0)

        if self.task_type == 'regression':
            return avg_y, np.squeeze(y_)
        return np.argmax(avg_y, axis=1)

    def _create_model(self):
        [x, y], [objectives, objectives_inference] = self._training()
        self._inference(x)
        params = [p for layer in self.layers for p in layer.params] + list(self.extra.itervalues())
        params_inf = [p for layerinf in self.layers_inf for p in layerinf.params] + list(self.extra_inf.itervalues())
        if self.task_type == 'regression':
            a1, a1i = T.exp(self.extra['a1']), T.exp(self.extra_inf['a1'])
            b1, b1i = T.exp(self.extra['b1']), T.exp(self.extra_inf['b1'])
            regpq, regpqi = - (1. / float(self.N)) * kldiv_gamma(a1, b1), - (1. / float(self.N_valid)) * kldiv_gamma(a1i, b1i)
            objectives.append(regpq)
            objectives_inference.append(regpqi)

        self.optimizer = self.alg_opt(objectives, objectives_inference, [x, y], params, params_inf, gradients=None,
                                      alpha=self.learning_rate, batch_size=self.batch_size, polyak=self.polyak,
                                      beta3=self.beta3, lr_decay=True, epsilon=nnu.eps, max_drop=self.ind_noise_lvl)

    def fit(self, xtrain, ytrain, xvalid=None, yvalid=None, verbose=False, print_every=1, sampling_rounds=1,
            xtest=None, ytest=None, llf=None, n_samples=2, return_best=False):
        rounding = lambda x: ['%.5f' % i for i in x]
        indices = range(self.N)
        nnu.prng.shuffle(indices)
        objective, objective_v = [], []
        train_errs, valid_errs, test_errs = [], [], []

        if self.task_type == 'regression':
            self.revy = lambda x: (x * np.std(ytrain, axis=0)) + np.mean(ytrain, axis=0)
        self._create_parameters()
        self._create_model()

        if xvalid is not None:
            # model selection according to a validation set
            eopt = np.inf
            best_layers = [copy(layer) for layer in self.layers_inf]
            best_extra = OrderedDict()
            for key, value in self.extra_inf.iteritems():
                if not value.get_value().shape:
                    best_extra[key] = theano.shared(value.get_value(borrow=False)[()], name=value.name + '_inf', borrow=False)
                    continue
                best_extra[key] = theano.shared(value.get_value(borrow=False), name=value.name + '_inf', borrow=False)

        for epoch in xrange(self.n_iter):
            t = time.time()
            if self.task_type == 'classification':
                yyy = ytrain[indices].astype(np.int32)
            else:
                yyy = ytrain[indices].astype(np.float32)
            inputs = [xtrain[indices].astype(np.float32), yyy]
            out = self.optimizer.train(inputs, verbose=verbose).tolist()
            loss, reg = out[0], out[1:]
            objective.append(loss + sum(reg))
            yp_train = self.predict(xtrain, samples=sampling_rounds)
            if self.task_type == 'classification':
                train_acc = (yp_train == ytrain).sum() / (1. * ytrain.shape[0])
            elif self.task_type == 'regression':
                train_acc = np.sqrt(np.mean(np.sum((ytrain - yp_train)**2, axis=1)))
            train_s = [loss + sum(reg), loss] + reg
            train_s += [100. * (1. - train_acc)] if self.task_type == 'classification' else [train_acc]
            train_errs.append(train_s[-1])
            if xvalid is not None:
                yp_valid = self.predict(xvalid, samples=sampling_rounds)
                if self.task_type == 'classification':
                    valid_acc = (yp_valid == yvalid).sum() / (1. * yvalid.shape[0])
                elif self.task_type == 'regression':
                    valid_acc = np.sqrt(np.mean(np.sum((yvalid - yp_valid)**2, axis=1)))
                verr = 100 * (1. - valid_acc) if self.task_type == 'classification' else valid_acc
                valid_errs.append(verr)
                if verr <= eopt:
                    eopt = verr
                    # store the parameters
                    for ii, layer in enumerate(self.layers_inf):
                        best_layers[ii].set_params(layer.params)
                    for key, value in self.extra_inf.iteritems():
                        best_extra[key].set_value(value.get_value(borrow=False), borrow=False)
            if xtest is not None:
                ypredt = self.predict(xtest, samples=sampling_rounds)
                test_err = 100 * (1. - (ypredt == ytest).sum() / (1. * ytest.shape[0]))
                test_errs.append(test_err)

            if (epoch + 1) % print_every == 0:
                string = 'Epoch ' + str(epoch + 1) + '/' + str(self.n_iter) + ', train: ' + str(rounding(train_s))
                if xvalid is not None:
                    string += ', valid: ' + str(rounding([verr]))

                if xtest is not None:
                    if self.task_type == 'regression':
                        ypred, sample_preds = self.predict(xtest, samples=n_samples)
                        rmse = np.sqrt(np.mean(np.sum((ytest - ypred)**2, axis=1)))
                        mean_ll = llf(self, sample_preds, ytest)
                        string += ', test: ' + str(rounding([rmse, mean_ll]))
                    elif self.task_type == 'classification':
                        string += ', test: ' + str(rounding([test_err]))

                dt = time.time() - t
                nnu.log_f(string + ', dt: ' + rounding([dt])[0], f=self.logtxt)

            nnu.prng.shuffle(indices)
        if xvalid is not None and return_best:
            for ii, layer in enumerate(best_layers):
                self.layers_inf[ii].set_params(layer.params)
            for key, value in best_extra.iteritems():
                self.extra_inf[key].set_value(value.get_value(borrow=False), borrow=False)
        return [objective, objective_v], [train_errs, valid_errs, test_errs]


