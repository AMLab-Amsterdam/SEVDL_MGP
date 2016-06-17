import gzip
import numpy as np
import cPickle as pkl
import nn_utils as nnu
import theano
from VMGNet import VMGNet
floatX = theano.config.floatX

f = gzip.open('data/mnist.pkl.gz', 'rb')
(xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = pkl.load(f)
f.close()

n_y = np.unique(ytrain).shape[0]
idx = nnu.prng.permutation(range(xtrain.shape[0]))
xtrain, ytrain = xtrain[idx], ytrain[idx]
xtrain, xtest = np.cast[floatX](xtrain), np.cast[floatX](xtest)

nn = VMGNet(xtrain.shape[0], xtrain.shape[1], n_y, batch_size=100, dimh=(150, 150, 150), n_iter=100,
            logtxt='vmgnet.txt', type_init='he2', n_inducing=50, ind_noise_lvl=0.01, task_type='classification')

output_nn = nn.fit(xtrain, ytrain, xvalid=xvalid, yvalid=yvalid, xtest=xtest, ytest=ytest, sampling_rounds=1)

preds = nn.predict(xtest, samples=1)
print 'Mean Test error:', 100. * ((preds != ytest).sum() / (1. * ytest.shape[0]))
