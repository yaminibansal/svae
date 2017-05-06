from __future__ import division, print_function
import matplotlib.pyplot as plt
import hickle as hkl
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam, sgd
from svae.svae import make_gradfun
from svae.nnet import init_gresnet, make_loglike, gaussian_mean, gaussian_info
from svae.models.vanilla_gmm import (run_inference, init_pgm_param, make_plotter_2d)
from svae.util import split_into_batches, get_num_datapoints
from autograd.util import flatten
from svae.distributions.gaussian import pack_dense
from toolz import curry

callback = lambda i, val, params, grad: print('{}: {}'.format(i, val))
flat = lambda struct: flatten(struct)[0]

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2*npr.permutation(np.einsum('ti,tij->tj', features, rotations))

@curry
def make_gradfun(run_inference, pgm_prior, data, batch_size, num_samples, natgrad_scale=1., callback=callback):
    _, unflat = flatten(pgm_prior)
    num_datapoints = get_num_datapoints(data)
    data_batches, num_batches = split_into_batches(data, batch_size)
    get_batch = lambda i: data_batches[i % num_batches]
    saved = lambda: None

    def mc_elbo(pgm_params, i):
        #Here nn_potentials are just the sufficient stats of the data
        x = get_batch(i)
        xxT = np.einsum('ij,ik->ijk', x, x)
        n = np.ones(x.shape[0]) if x.ndim == 2 else 1.
        nn_potentials = pack_dense(xxT, x, n, n)
        saved.stats, global_kl, local_kl  = run_inference(pgm_prior, pgm_params, nn_potentials)
        return (- global_kl - num_batches * local_kl) / num_datapoints #CHECK

    def gradfun(params, i):
        pgm_params = params
        val = -mc_elbo(pgm_params,i)
        pgm_natgrad = -natgrad_scale / num_datapoints * \
                      (flat(pgm_prior) + num_batches*flat(saved.stats) - flat(pgm_params))
        #print(flat(pgm_prior), num_batches*flat(saved.stats), -flat(pgm_params))
        grad = unflat(pgm_natgrad)
        if callback: callback(i, val, params, grad)
        return grad

    return gradfun
    
    

if __name__ == "__main__":
    #npr.seed(1)
    seed_no = npr.randint(1000)
    print(seed_no)
    npr.seed(seed_no)
    plt.ion()
    plt.autoscale(False)

    num_clusters = 3           # number of clusters in pinwheel data
    samples_per_cluster = 100  # number of samples per cluster in pinwheel
    K = 100                    # number of components in mixture model
    N = 2                      # number of latent dimensions
    P = 2                      # number of observation dimensions
    num_iters = 10000
    plot_every = 100
    pred_ll = np.zeros(num_iters/plot_every)

    # generate/load synthetic data
    #data = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)
    filename = '/Users/ybansal/Documents/PhD/Courses/CS282/Project/Code/Data/pinwheel_data.hkl'
    file = open(filename, 'r')
    storage_reloaded = hkl.load(file)
    #data = hkl.load(file)
    file.close()
    data = storage_reloaded['data']
    x_test = data

    # set prior natparam to something sparsifying but otherwise generic
    pgm_prior_params = init_pgm_param(K, N, alpha=0.05/K, niw_conc=.5)

    # initialize gmm parameters
    pgm_params = init_pgm_param(K, N, alpha=1., niw_conc=1., random_scale=2.)
    params = pgm_params

    # set up encoder/decoder and plotting
    plot = make_plotter_2d(data, pred_ll, x_test, num_clusters, params, plot_every=plot_every)

    # instantiate svae gradient function
    gradfun = make_gradfun(run_inference, pgm_prior_params, data)

    # optimize
    params = sgd(gradfun(batch_size=50, num_samples=1, natgrad_scale=1, callback=plot),
                  params, num_iters=num_iters)

