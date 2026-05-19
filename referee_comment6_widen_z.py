"""
Front/back degeneracy test for referee Comment 6.

Re-fits an example stream with the z0 prior widened from the positive
half-normal N+(0, 150) to the full normal N(0, 150) so that z0 can take
negative values. With projected data only, this should reveal two
symmetric posterior modes at +z0 and -z0, while leaving the marginal
posteriors on all halo parameters (including q) unchanged.

Uses seed 85 (|z0| ~ 180 kpc) rather than the paper's seed 46 (|z0| ~ 38
kpc) so that the +z/-z mirror modes are well separated and visually
obvious. q is constrained to [0.5, 1.5] to match the prior used in the
revised paper.

Usage on the cluster:
    python referee_comment6_widen_z.py
"""

import os
import pickle
import numpy as np
import multiprocessing as mp

import jax
import jax.numpy as jnp

import dynesty
import dynesty.utils as dyut

from likelihoods import BAD_VAL
from spray_base import generate_stream_spray_base
from utils import get_q, get_track

import corner
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


# The canonical log_likelihood_spray_base calls get_q with its default
# q_max=2.0. The referee asked for q in [0.5, 1.5], so we wrap the same
# pipeline locally with q_max=1.5 rather than mutating the shared helper.
Q_MIN = 0.5
Q_MAX = 1.5


def log_likelihood_spray_base_q15(params, dict_data, seed=13, min_count=100):
    q = get_q(params[2], params[3], params[4], q_min=Q_MIN, q_max=Q_MAX)
    params = np.concatenate([params[:2], [q], params[2:8], [0.], params[8:], [1.]])

    theta_stream, xv_stream, _, _ = generate_stream_spray_base(params, seed)
    _, _, r_bin, _ = get_track(theta_stream, xv_stream[:, 0], xv_stream[:, 1])

    arg_take = ~np.isnan(dict_data['r_bin']) * (dict_data['count'] > min_count)
    n_bad = np.sum(np.isnan(r_bin[arg_take]))

    if np.all(np.isnan(r_bin)):
        return BAD_VAL * len(r_bin)
    if n_bad == 0:
        return -0.5 * np.sum(
            ((r_bin[arg_take] - dict_data['r_bin'][arg_take]) /
             dict_data['r_sig'][arg_take])**2
        )
    return BAD_VAL * n_bad


def prior_transform_widez(p):
    """
    Identical to priors.prior_transform, except z0 is sampled from the
    *full* N(0, 150) instead of the positive half N+(0, 150). All other
    parameters are unchanged.
    """
    (logM, Rs, dirx, diry, dirz,
     logm, rs,
     x0, z0, vx0, vy0, vz0,
     t0) = p

    logM1 = 11 + 3 * logM
    Rs1   = 5 + 20 * Rs

    dirx1 = jax.scipy.special.ndtri(dirx)
    diry1 = jax.scipy.special.ndtri(diry)
    dirz1 = jax.scipy.special.ndtri(0.5 + dirz / 2)

    logm1 = 7 + 2 * logm
    rs1   = 1 + 2 * rs

    x1 = jax.scipy.special.ndtri(0.5 + x0 / 2) * 150
    # Widened prior on z0: full N(0, 150)
    z1 = jax.scipy.special.ndtri(z0) * 150

    vx1 = jax.scipy.special.ndtri(vx0) * 250
    vy1 = jax.scipy.special.ndtri(0.5 + vy0 / 2) * 250
    vz1 = jax.scipy.special.ndtri(vz0) * 250

    t1 = 1 + 3 * t0

    return jnp.array([
        logM1, Rs1, dirx1, diry1, dirz1,
        logm1, rs1,
        x1, z1, vx1, vy1, vz1,
        t1,
    ])


def dynesty_fit(dict_data, ndim=13, nlive=2000, sigma=2):
    # NB: in individual_fits.py the second positional arg passed via logl_args
    # is bound to `seed` in log_likelihood_spray_base, so the baseline runs
    # used spray seed = sigma. We replicate that here for comparability.
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(
            log_likelihood_spray_base_q15,
            prior_transform_widez,
            ndim,
            logl_args=(dict_data, sigma),
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
    seed = 85
    ndim = 13
    nlive = 2000
    sigma = 2  # standard 2% noise, matches the paper example

    path = f'/data/dc824-2/MockStreams/seed{seed}'

    with open(os.path.join(path, 'dict_stream.pkl'), 'rb') as f:
        dict_data = pickle.load(f)

    # Same noise realisation as the baseline fit for this seed
    params_data = dict_data['params']
    params_data = np.concatenate([params_data[:2], params_data[3:9], params_data[10:-1]])

    r_sig = dict_data['r_bin'] * sigma / 100
    rng = np.random.default_rng(int(seed))
    r_err = rng.normal(0, r_sig)

    dict_data['r_bin'] += r_err
    dict_data['r_sig'] = r_sig
    dict_data['x_bin'] = dict_data['r_bin'] * np.cos(dict_data['theta_bin'])
    dict_data['y_bin'] = dict_data['r_bin'] * np.sin(dict_data['theta_bin'])

    print(f'Refitting seed {seed} with widened z0 prior (full N(0,150)) ...')
    dict_results = dynesty_fit(dict_data, ndim=ndim, nlive=nlive, sigma=sigma)

    out_pkl = os.path.join(path,
        f'dict_results_widez_nlive{nlive}_sigma{sigma}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(dict_results, f)
    print('Saved:', out_pkl)

    # Corner plot with the true value marked
    labels = ['logM', 'Rs', 'dirx', 'diry', 'dirz',
              'logm', 'rs',
              'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time']
    fig = corner.corner(dict_results['samps'],
                        labels=labels,
                        color='blue',
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 16},
                        truths=params_data,
                        truth_color='red')
    fig.savefig(os.path.join(path,
        f'corner_widez_nlive{nlive}_sigma{sigma}.pdf'),
        bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Focused 1D plot on z0 to highlight the symmetric front/back modes
    z0_samps = dict_results['samps'][:, 8]
    plt.figure(figsize=(8, 6))
    plt.hist(z0_samps, bins=60, density=True, alpha=0.7, color='blue')
    plt.axvline(params_data[8], color='red', linestyle='--', lw=2,
                label=f'True $z_0 = {params_data[8]:.1f}$')
    plt.axvline(-params_data[8], color='red', linestyle=':', lw=2,
                label=f'Mirror $-z_0 = {-params_data[8]:.1f}$')
    plt.xlabel(r'$z_0$ [kpc]')
    plt.ylabel('Density')
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(path,
        f'z0_posterior_widez_nlive{nlive}_sigma{sigma}.pdf'),
        bbox_inches='tight', dpi=300)
    plt.close()

    # q posterior, for comparison with the baseline fit
    q_samps = np.asarray(get_q(dict_results['samps'][:, 2],
                               dict_results['samps'][:, 3],
                               dict_results['samps'][:, 4],
                               q_min=Q_MIN, q_max=Q_MAX))
    plt.figure(figsize=(8, 6))
    plt.hist(q_samps, bins=30, density=True, alpha=0.7, color='blue')
    plt.axvline(dict_data['params'][2], color='red', linestyle='--', lw=2,
                label=f'True $q = {dict_data["params"][2]:.2f}$')
    plt.xlabel(r'Halo flattening $q$')
    plt.ylabel('Density')
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(path,
        f'q_posterior_widez_nlive{nlive}_sigma{sigma}.pdf'),
        bbox_inches='tight', dpi=300)
    plt.close()

    print('Done.')
