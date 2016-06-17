import numpy as np
import sklearn.cross_validation as cv
from VMGNet import VMGNet
from sklearn.preprocessing import StandardScaler
from scipy.misc import logsumexp
from scipy.special import psi


data = np.loadtxt('data/yacht_hydrodynamics.data.txt')
x, y = data[:, :-1], data[:, -1]
y = y.reshape((y.shape[0], 1))


def loglike(nn, sample_preds, y):
    """Return the Avg. Test Log-Likelihood
    """
    if y.shape[1] == 1:
        y = y.ravel()
    sample_ll = np.zeros((sample_preds.shape[1], sample_preds.shape[0]))
    a, b = np.exp(nn.extra_inf['a1'].get_value()), np.exp(nn.extra_inf['b1'].get_value())
    etau, elogtau = (a / b).astype(np.float32), (psi(a) - np.log(b)).astype(np.float32)
    for sample in xrange(sample_preds.shape[0]):
        ypred = sample_preds[sample].astype(np.float32)
        if len(y.shape) > 1:
            sll = -.5 * np.sum(etau * (y - ypred)**2, axis=1)
        else:
            sll = -.5 * etau * (y - ypred)**2
        sample_ll[:, sample] = sll

    return np.mean(logsumexp(sample_ll, axis=1) - np.log(sample_preds.shape[0]) - .5 * np.log(2*np.pi) + .5 * elogtau.sum())


xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, train_size=0.9, random_state=1)
std_scx = StandardScaler().fit(xtrain)
xtrain, xtest = std_scx.transform(xtrain), std_scx.transform(xtest)
nn = VMGNet(xtrain.shape[0], xtrain.shape[1], ytrain.shape[1], batch_size=5, dimh=(50,), n_iter=2000,
            logtxt='vmgnet.txt', seed=3, task_type='regression', sampling_pred=True, type_init='he2', n_inducing=5,
            ind_noise_lvl=0.005)
_, _ = nn.fit(xtrain.astype(np.float32), ytrain.astype(np.float32), xtest=xtest, ytest=ytest, print_every=200,
              llf=loglike, n_samples=2)

ypred, sample_preds = nn.predict(xtest, samples=100)
rmse = np.sqrt(np.mean(np.sum((ytest - ypred) ** 2, axis=1)))
mean_ll = loglike(nn, sample_preds, ytest)
print 'RMSE:', rmse, 'mean_ll:', mean_ll