from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import digamma, gammaln

def expectedstats(natparam):
    #Returns E_{q(v)}[\eta_z(V)] where \eta_z(V)_i = [ln(V_i), ln(1-V_i)]
    #natparam size is (T-1)x2
    alpha_beta = natparam + 1
    return digamma(alpha_beta) - digamma(np.sum(alpha_beta, axis=1, keepdims=True))

def logZ(natparam):
    alpha_beta = natparam + 1
    return np.sum(np.sum(gammaln(alpha_beta)) - gammaln(np.sum(alpha_beta, axis=1)))

def var_expectedstats(natparam):
    # Returns E_{q(v)}[\eta_z(V)] where \eta_z(V)_i = ln(V_i) + \sum_{j < i} ln(1-V_j)
    # q is truncated to level T with q(v_T) = 1 but p is not truncated, tho terms > T don't really feature in the calculations
    # natparam shape is (T-1)x2, refers to \gamma_{t,0/1}
    # Returned function shape will be T

    return np.append(np.array([ digamma(natparam[i, 0]) - digamma(natparam[i,0]+natparam[i,1])\
                                + np.sum(np.array([digamma(natparam[j, 1]) - digamma(natparam[j,0]+natparam[j,1]) for j in range(i)])) for i in range(natparam.shape[0])]), (np.sum(digamma(natparam[:,1])-digamma(natparam[:,0]+natparam[:,1]))))
            
