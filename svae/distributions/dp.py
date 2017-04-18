from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import digamma, gammaln

def expectedstats(natparam):
    #natparam size is (T-1)x2
    alpha_beta = natparam + 1
    return digamma(alpha_beta) - digamma(np.sum(alpha_beta, axis=1, keepdims=True))

def logZ(natparam):
    alpha_beta = natparam + 1
    return np.sum(np.sum(gammaln(alpha_beta)) - gammaln(np.sum(alpha_beta, axis=1)))

def var_expectedstats(natparam):
    # natparam shape is (T-1)x2, refers to \gamma
    # Returned function shape should be (T) so append 0

    return np.array([ digamma(natparam[i, 0]) - digamma(natparam[i,0]+natparam[i,1])\
                                + np.sum(np.array([digamma(natparam[j, 1]) - digamma(natparam[j,0]+natparam[j,1]) for j in range(i)])) for i in range(natparam.shape[0])])
            
