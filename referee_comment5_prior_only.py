"""
Prior-only test for referee Comment 5.

Runs Dynesty with a constant log-likelihood, so the posterior is the prior.
Transforms the (dirx, diry, dirz) samples through get_q and checks that the
implied posterior on q is uniform on [q_min, q_max].

Usage on the cluster:
    python referee_comment5_prior_only.py
"""

import os
import pickle
import numpy as np
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

from priors import prior_transform
from utils import get_q

import corner
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


def constant_loglike(params):
    # Flat likelihood -> posterior == prior.
    return 0.0


def dynesty_fit_prior(ndim=13, nlive=2000):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(
            constant_loglike,
            prior_transform,
            ndim,
            nlive=nlive,
            sample='rslice',
            pool=poo,
            queue_size=nthreads * 2,
        )
        dns.run_nested(n_effective=10000)

    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl = res.logl[inds]

    return {
        'dns': dns,
        'samps': samps,
        'logl': logl,
        'logz': res.logz,
        'logzerr': res.logzerr,
    }


if __name__ == "__main__":
    ndim = 13
    nlive = 2000

    out_dir = '/data/dc824-2/MockStreams/referee_comment5'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Running prior-only Dynesty with nlive={nlive} ...')
    dict_results = dynesty_fit_prior(ndim=ndim, nlive=nlive)

    with open(os.path.join(out_dir, f'dict_results_prior_only_nlive{nlive}.pkl'), 'wb') as f:
        pickle.dump(dict_results, f)

    samps = dict_results['samps']

    # Corner plot of the parameters (should reproduce the priors)
    labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz',
              'logm', 'rs',
              'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time']
    fig = corner.corner(samps,
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 16})
    fig.savefig(os.path.join(out_dir, f'corner_prior_only_nlive{nlive}.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Map dirx, diry, dirz -> q  (should be uniform on [q_min, q_max])
    q_min, q_max = 0.5, 1.5
    q_samps = np.asarray(get_q(samps[:, 2], samps[:, 3], samps[:, 4],
                               q_min=q_min, q_max=q_max))

    plt.figure(figsize=(8, 6))
    plt.hist(q_samps, bins=40, density=True, alpha=0.7,
             color='blue', range=(q_min, q_max))
    plt.axhline(1.0 / (q_max - q_min), color='red', linestyle='--', lw=2)
    plt.xlabel(r'Halo flattening $q$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'q_prior_only_nlive{nlive}.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print('Done. Outputs in', out_dir)
