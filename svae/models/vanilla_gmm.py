from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from itertools import repeat
from functools import partial
from scipy.stats import norm

from svae.util import unbox, getval, flat, normalize
from svae.distributions import dirichlet, categorical, niw, gaussian

from matplotlib.colors import LinearSegmentedColormap
gridsize=75
colors = np.array([
[106,61,154],  # Dark colors
[31,120,180],
[51,160,44],
[227,26,28],
[255,127,0],
[166,206,227],  # Light colors
[178,223,138],
[251,154,153],
[253,191,111],
[202,178,214],
]) / 256.0

### inference functions for the SVAE interface

def run_inference(prior_natparam, global_natparam, nn_potentials):
    stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return unbox(stats), global_kl, local_kl

### GMM prior on \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)

def init_pgm_param(K, N, alpha, niw_conc=10., random_scale=0.):
    def init_niw_natparam(N):
        nu, S, m, kappa = N+niw_conc, (N+niw_conc)*np.eye(N), np.zeros(N), niw_conc
        m = m + random_scale * npr.randn(*m.shape)
        return niw.standard_to_natural(S, m, kappa, nu)

    dirichlet_natparam = alpha * (npr.rand(K) if random_scale else np.ones(K))
    niw_natparam = np.stack([init_niw_natparam(N) for _ in range(K)])

    return dirichlet_natparam, niw_natparam

def prior_logZ(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    return dirichlet.logZ(dirichlet_natparam) + niw.logZ(niw_natparams)

def prior_expectedstats(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    dirichlet_expectedstats = dirichlet.expectedstats(dirichlet_natparam)
    niw_expectedstats = niw.expectedstats(niw_natparams)
    return dirichlet_expectedstats, niw_expectedstats

def prior_kl(global_natparam, prior_natparam):
    expected_stats = flat(prior_expectedstats(global_natparam))
    natparam_difference = flat(global_natparam) - flat(prior_natparam)
    logZ_difference = prior_logZ(global_natparam) - prior_logZ(prior_natparam)
    return np.dot(natparam_difference, expected_stats) - logZ_difference

### GMM mean field functions

def local_meanfield(global_natparam, gaussian_suff_stats):
    # global_natparam = \eta_{\theta}^0
    dirichlet_natparam, niw_natparams = global_natparam
    #node_potentials = gaussian.pack_dense(*node_potentials)

    #### compute expected global parameters using current global factors
    # label_global = E_{q(\pi)}[t(\pi)] here q(\pi) is posterior which is dirichlet with parameter dirichlet_natparam and t is [log\pi_1, log\pi_2....]
    # gaussian_globals = E_{q(\mu, \Sigma)}[t(\mu, \Sigma)] here q(\mu, \Sigma) is posterior which is NIW
    # label_stats = E_{q(z)}[t(z)] -> categorical expected statistics. Shape = (batch_size, K)
    # gaussian_suff_stats  Shape = (batch_size, 4, 4)
    label_global = dirichlet.expectedstats(dirichlet_natparam)
    gaussian_globals = niw.expectedstats(niw_natparams)

    #### compute values that depend directly on boxed node_potentials at optimum
    label_natparam, label_stats, label_kl = \
        label_meanfield(label_global, gaussian_globals, gaussian_suff_stats)

    #### collect sufficient statistics for gmm prior (sum across conditional iid)
    dirichlet_stats = np.sum(label_stats, 0)
    niw_stats = np.tensordot(label_stats, gaussian_suff_stats, [0, 0])

    local_stats = label_stats, gaussian_suff_stats
    prior_stats = dirichlet_stats, niw_stats
    natparam = label_natparam
    kl = label_kl 

    return prior_stats, natparam, kl

def label_meanfield(label_global, gaussian_globals, gaussian_suff_stats):
    # Ref. Eq 39
    # label_global = E_{q(\pi)}[t(\pi)] where q(\pi) is dirichlet and t(\pi) is {log\pi_i}
    # stats = E_{q(z)}[t(z)] -> categorical expected statistics
    # gaussian_suff_stats = t(x) where t(x) is [x, xxT] Shape = (batch_size, 4, 4)
    # gaussian_globals = niw expected stats (Shape = (K, 4, 4))
    # node_potenials, label_global, natparam Shape = (batch_size, K)
    
    node_potentials = np.tensordot(gaussian_suff_stats, gaussian_globals, [[1,2], [1,2]])
    natparam = node_potentials + label_global
    stats = categorical.expectedstats(natparam)
    kl = np.tensordot(stats, node_potentials) - categorical.logZ(natparam)
    return natparam, stats, kl

### plotting util for 2D

def make_plotter_2d(data, num_clusters, params, plot_every):
    
    import matplotlib.pyplot as plt
    if data.shape[1] != 2: raise ValueError, 'make_plotter_2d only works with 2D data'

    fig, observation_axis = plt.subplots(1, 1, figsize=(8,4))

    observation_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
    observation_axis.set_aspect('equal')
    observation_axis.autoscale(False)
    #observation_axis.axis('off')
    fig.tight_layout()

    def plot_ellipse(ax, alpha, mean, cov, line=None):
        t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
        if line:
            line.set_data(ellipse)
            line.set_alpha(alpha)
        else:
            ax.plot(ellipse[0], ellipse[1], alpha=alpha, linestyle='-', linewidth=2)

    def get_component(niw_natparam):
        neghalfJ, h, _, _ = gaussian.unpack_dense(niw_natparam)
        J = -2 * neghalfJ
        return np.linalg.solve(J, h), np.linalg.inv(J)

    def plot_components(ax, params):
        pgm_params = params
        dirichlet_natparams, niw_natparams = pgm_params
        normalize = lambda arr: np.minimum(1., arr / np.sum(arr) * num_clusters)
        weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
        components = map(get_component, niw.expectedstats(niw_natparams))
        S, m, kappa, nu = niw.natural_to_standard(niw_natparams)
        #print m
        lines = repeat(None) if isinstance(ax, plt.Axes) else ax
        for weight, (mu, Sigma), line in zip(weights, components, lines):
            #print mu
            plot_ellipse(ax, weight, mu, Sigma, line)


    def plot(i, val, params, grad):
        print('{}: {}'.format(i, val))
        if (i % plot_every) == (-1 % plot_every):
            plot_components(observation_axis.lines[1:], params)
            plt.pause(0.1)

    plot_components(observation_axis, params)
    plt.pause(0.1)

    return plot
