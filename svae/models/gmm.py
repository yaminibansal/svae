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

def run_inference(prior_natparam, global_natparam, nn_potentials, num_samples):
    _, stats, local_natparam, local_kl = local_meanfield(global_natparam, nn_potentials)
    samples = gaussian.natural_sample(local_natparam[1], num_samples)
    global_kl = prior_kl(global_natparam, prior_natparam)
    return samples, unbox(stats), global_kl, local_kl

def make_encoder_decoder(recognize, decode):
    def encode_mean(data, natparam, recogn_params):
        nn_potentials = recognize(recogn_params, data)
        (_, gaussian_stats), _, _, _ = local_meanfield(natparam, nn_potentials)
        _, Ex, _, _ = gaussian.unpack_dense(gaussian_stats)
        return Ex

    def decode_mean(z, phi):
        mu, _ = decode(phi, z)
        return mu.mean(axis=1)

    return encode_mean, decode_mean

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

def local_meanfield(global_natparam, node_potentials):
    # global_natparam = \eta_{\theta}^0
    # node_potentials = r(\phi, y)
    
    dirichlet_natparam, niw_natparams = global_natparam
    node_potentials = gaussian.pack_dense(*node_potentials)

    #### compute expected global parameters using current global factors
    # label_global = E_{q(\pi)}[t(\pi)] here q(\pi) is posterior which is dirichlet with parameter dirichlet_natparam and t is [log\pi_1, log\pi_2....]
    # gaussian_globals = E_{q(\mu, \Sigma)}[t(\mu, \Sigma)] here q(\mu, \Sigma) is posterior which is NIW    
    label_global = dirichlet.expectedstats(dirichlet_natparam)
    gaussian_globals = niw.expectedstats(niw_natparams)

    #### compute mean field fixed point using unboxed node_potentials
    label_stats = meanfield_fixed_point(label_global, gaussian_globals, getval(node_potentials))

    #### compute values that depend directly on boxed node_potentials at optimum
    gaussian_natparam, gaussian_stats, gaussian_kl = \
        gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
    label_natparam, label_stats, label_kl = \
        label_meanfield(label_global, gaussian_globals, gaussian_stats)

    #### collect sufficient statistics for gmm prior (sum across conditional iid)
    dirichlet_stats = np.sum(label_stats, 0)
    niw_stats = np.tensordot(label_stats, gaussian_stats, [0, 0])

    local_stats = label_stats, gaussian_stats
    prior_stats = dirichlet_stats, niw_stats
    natparam = label_natparam, gaussian_natparam
    kl = label_kl + gaussian_kl

    return local_stats, prior_stats, natparam, kl

def meanfield_fixed_point(label_global, gaussian_globals, node_potentials, tol=1e-3, max_iter=100):
    kl = np.inf
    label_stats = initialize_meanfield(label_global, node_potentials)
    for i in xrange(max_iter):
        gaussian_natparam, gaussian_stats, gaussian_kl = \
            gaussian_meanfield(gaussian_globals, node_potentials, label_stats)
        label_natparam, label_stats, label_kl = \
            label_meanfield(label_global, gaussian_globals, gaussian_stats)

        # recompute gaussian_kl linear term with new label_stats b/c labels were updated
        gaussian_global_potentials = np.tensordot(label_stats, gaussian_globals, [1, 0])
        linear_difference = gaussian_natparam - gaussian_global_potentials - node_potentials
        gaussian_kl = gaussian_kl + np.tensordot(linear_difference, gaussian_stats, 3)

        kl, prev_kl = label_kl + gaussian_kl, kl
        if abs(kl - prev_kl) < tol:
            break
    else:
        print 'iteration limit reached'

    return label_stats

def gaussian_meanfield(gaussian_globals, node_potentials, label_stats):
    # Ref. Eq 39
    # gaussian_globals = E_{q(\mu, \Sigma)}[t(\mu, \Sigma)] here q(\mu, \Sigma) is posterior which is NIW. Shape = (K, 4, 4)
    # label_stats = E_{q(z)}[t(z)] -> categorical expected statistics. Shape = (batch_size, K)
    # stats = E_{q(z)}[t(z)] -> Gaussian expected statistics Shape = (batch_size, 4, 4)
    # node_potentials = r(\phi, y) Shape = (batch_size, 4, 4)
    #print gaussian_globals.shape, node_potentials.shape, label_stats.shape
    global_potentials = np.tensordot(label_stats, gaussian_globals, [1, 0])
    natparam = node_potentials + global_potentials #using Eq. 39
    stats = gaussian.expectedstats(natparam)
    #print stats.shape
    kl = np.tensordot(node_potentials, stats, 3) - gaussian.logZ(natparam)
    return natparam, stats, kl

def label_meanfield(label_global, gaussian_globals, gaussian_stats):
    # Ref. Eq 39
    # label_global = E_{q(\pi)}[t(\pi)] where q(\pi) is dirichlet and t(\pi) is {log\pi_i}
    # stats = E_{q(z)}[t(z)] -> categorical expected statistics
    # gaussian_stats = E_{q(x)}[t(x)] where q(x) is NIW and t(x) is [x, xxT]
    # gaussian_globals = \eta_x^0(\theta)
    
    node_potentials = np.tensordot(gaussian_stats, gaussian_globals, [[1,2], [1,2]])
    natparam = node_potentials + label_global
    stats = categorical.expectedstats(natparam)
    kl = np.tensordot(stats, node_potentials) - categorical.logZ(natparam)
    return natparam, stats, kl

def initialize_meanfield(label_global, node_potentials):
    # K is the number of mixture components
    # I think T is the batch size 
    T, K = node_potentials.shape[0], label_global.shape[0]
    return normalize(npr.rand(T, K))

### plotting util for 2D

def make_plotter_2d(recognize, decode, data, num_clusters, params, plot_every):
    import matplotlib.pyplot as plt
    if data.shape[1] != 2: raise ValueError, 'make_plotter_2d only works with 2D data'

    fig, (observation_axis, latent_axis, density_axis) = plt.subplots(1, 3, figsize=(8,4))
    #fig2, elbo_axis = plt.subplots(1, 1, figsize=(8,4))
    encode_mean, decode_mean = make_encoder_decoder(recognize, decode)

    observation_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
    observation_axis.set_aspect('equal')
    observation_axis.autoscale(False)
    #density_axis.autoscale(False)
    latent_axis.set_aspect('equal')
    density_axis.set_aspect('equal')
    observation_axis.axis('off')
    latent_axis.axis('off')
    density_axis.axis('off')    
    fig.tight_layout()

    def plot_encoded_means(ax, params):
        pgm_params, loglike_params, recogn_params = params
        encoded_means = encode_mean(data, pgm_params, recogn_params)
        if isinstance(ax, plt.Axes):
            ax.plot(encoded_means[:,0], encoded_means[:,1], color='k', marker='.', linestyle='')
        elif isinstance(ax, plt.Line2D):
            ax.set_data(encoded_means.T)
        else:
            raise ValueError

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



    def get_hexbin_coords(ax, xlims, ylims, gridsize):
        coords = ax.hexbin([], [], gridsize=gridsize, extent=tuple(xlims)+tuple(ylims)).get_offsets()
        del ax.collections[-1]
        return coords

    def plot_transparent_hexbin(ax, func, xlims, ylims, gridsize, color):
        cdict = {'red':   ((0., color[0], color[0]), (1., color[0], color[0])),
                 'green': ((0., color[1], color[1]), (1., color[1], color[1])),
                 'blue':  ((0., color[2], color[2]), (1., color[2], color[2])),
                 'alpha': ((0., 0., 0.), (1., 1., 1.))}

        new_cmap = LinearSegmentedColormap('Custom', cdict)
        plt.register_cmap(cmap=new_cmap)

        coords = get_hexbin_coords(ax, xlims, ylims, gridsize)
        c = func(coords)

        x, y = coords.T

        ax.hexbin(x.ravel(), y.ravel(), c.ravel(),
                  cmap=new_cmap, linewidths=0., edgecolors='none',
                  gridsize=gridsize, vmin=0., vmax=1., zorder=1)

    def decode_density(latent_locations, phi, decode, weight=1.):
        mu, sigmasq = decode(phi, latent_locations)
        #sigmasq = np.exp(log_sigmasq)

        mu = mu if mu.ndim == 3 else mu[:,None,:]
        sigmasq = sigmasq if sigmasq.ndim == 3 else sigmasq[:,None,:]

        def density(r):
            distances = np.sqrt(((r[None,:,:] - mu)**2 / sigmasq).sum(2))
            return weight * (norm.pdf(distances) / np.sqrt(sigmasq).prod(2)).mean(0)

        return density

    def plot_components(ax, params):
        pgm_params, loglike_params, recogn_params = params
        dirichlet_natparams, niw_natparams = pgm_params
        normalize = lambda arr: np.minimum(1., arr / np.sum(arr) * num_clusters)
        weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
        components = map(get_component, niw.expectedstats(niw_natparams))
        num_samples = 200
        lines = repeat(None) if isinstance(ax, plt.Axes) else ax
        for weight, (mu, Sigma), line in zip(weights, components, lines):
            plot_ellipse(ax, weight, mu, Sigma, line)


    def plot_density(density_ax, params):
        pgm_params, loglike_params, recogn_params = params
        dirichlet_natparams, niw_natparams = pgm_params
        normalize = lambda arr: np.minimum(1., arr / np.sum(arr) * num_clusters)
        weights = normalize(np.exp(dirichlet.expectedstats(dirichlet_natparams)))
        components = map(get_component, niw.expectedstats(niw_natparams))
        num_samples = 1000
        #lines = repeat(None) if isinstance(ax, plt.Axes) else ax
        idx = 0
        for weight, (mu, Sigma) in zip(weights, components):
            samples = npr.RandomState(0).multivariate_normal(mu, Sigma, num_samples)
            density = decode_density(samples, loglike_params, decode, 75. * weight)
            density_axis.plot(data[:,0], data[:,1], color='k', marker='.', linestyle='')
            xlim, ylim = density_axis.get_xlim(), density_axis.get_ylim()
            plot_transparent_hexbin(density_axis, density, xlim, ylim, gridsize, colors[idx % len(colors)])
            idx+=1


    def plot(i, val, params, grad):
        print('{}: {}'.format(i, val))
        if (i % plot_every) == (-1 % plot_every):
            plot_encoded_means(latent_axis.lines[0], params)
            plot_components(latent_axis.lines[1:], params)
            density_axis.cla()
            density_axis.set_aspect('equal')
            density_axis.axis('off')
            plot_density(density_axis, params)
            plt.title('ELBO = '+str(-val))
            plt.pause(0.1)
            plt.savefig(str(i)+'pinwheel.png')

    plot_encoded_means(latent_axis, params)
    plot_components(latent_axis, params)
    plot_density(density_axis, params)
    plt.pause(0.1)
    plt.savefig('star_pinwheel.png')

    return plot
