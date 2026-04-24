"""
Population fit model mismatch test — for referee Comment 15.

True underlying distribution is bimodal (two Gaussians), fitted with a
single Gaussian.  Demonstrates what happens when the population model
does not match the true distribution.

Usage:
    python population_fits_mismatch.py \
        --path /data/dc824-2/MockStreams \
        --true_mu1 0.8 --true_mu2 1.2 \
        --true_sigma1 0.1 --true_sigma2 0.1 \
        --fit_dist gaussian \
        --N_pop 50 \
        --nlive_pop 500
"""

import os
import argparse
import pickle
import numpy as np
import corner
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import dynesty
import dynesty.utils as dyut

from population_fits import (
    log_likelihood,
    prior_transform_gaussian,
    prior_transform_uniform,
    prior_transform_binomial,
    subset_as_binomial,
)
from utils import get_q


class DummyClass:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return DummyClass()
    def __call__(self, *args, **kwargs):
        return DummyClass()
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return DummyClass


def safe_load(filepath):
    with open(filepath, 'rb') as f:
        return SafeUnpickler(f).load()


def load_all_streams(path, nlive_ind, sigma_noise):
    """Load individual fit results and true q values for all available seeds."""
    q_true_all, q_fits_all = [], []
    seed = 0
    while True:
        path_seed = os.path.join(path, f'seed{seed}')
        res_file = os.path.join(path_seed, f'dict_results_nlive{nlive_ind}_sigma{sigma_noise}.pkl')
        stream_file = os.path.join(path_seed, 'dict_stream.pkl')
        if not os.path.exists(res_file):
            if seed > 200:
                break
            seed += 1
            continue
        dict_results = safe_load(res_file)
        dict_stream = safe_load(stream_file)
        q_fits_all.append(np.asarray(get_q(*dict_results['samps'][:, 2:5].T)))
        q_true_all.append(float(dict_stream['params'][2]))
        seed += 1
    q_true_all = np.array(q_true_all)
    print(f'Loaded {len(q_true_all)} individual fits from {path}')
    return q_true_all, q_fits_all


def main():
    parser = argparse.ArgumentParser(description='Population fit model mismatch test')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to MockStreams directory')
    parser.add_argument('--true_mu1', type=float, default=0.8,
                        help='Mean of first Gaussian component')
    parser.add_argument('--true_mu2', type=float, default=1.2,
                        help='Mean of second Gaussian component')
    parser.add_argument('--true_sigma1', type=float, default=0.1,
                        help='Sigma of first Gaussian component')
    parser.add_argument('--true_sigma2', type=float, default=0.1,
                        help='Sigma of second Gaussian component')
    parser.add_argument('--fit_dist', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'binomial'],
                        help='Population model to fit (default: gaussian = mismatch)')
    parser.add_argument('--N_pop', type=int, default=50,
                        help='Number of streams in the population')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='Mixing weight for the first (oblate) component (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subset selection')
    parser.add_argument('--nlive_ind', type=int, default=2000,
                        help='nlive used in individual fits (for loading)')
    parser.add_argument('--sigma_noise', type=int, default=2,
                        help='Sigma label used in individual fits (for loading)')
    parser.add_argument('--nlive_pop', type=int, default=500,
                        help='nlive for population fits')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle path')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.path,
            f'pop_mismatch_BtoG_N{args.N_pop}_prob{args.prob}_seed{args.seed}.pkl')

    # Load all individual fits once
    q_true_all, q_fits_all = load_all_streams(args.path, args.nlive_ind, args.sigma_noise)

    if args.fit_dist == 'gaussian':
        ndim = 2
        labels = [r'$\mu$', r'$\sigma$']
    elif args.fit_dist == 'uniform':
        ndim = 2
        labels = [r'$a$', r'$\delta$']
    elif args.fit_dist == 'binomial':
        ndim = 5
        labels = [r'$\mu_1$', r'$\mu_2$', r'$\sigma_1$', r'$\sigma_2$', r'$p$']

    # Select bimodal subset using mixture weights
    idx = subset_as_binomial(
        q_true_all,
        mu1=args.true_mu1, mu2=args.true_mu2,
        sigma1=args.true_sigma1, sigma2=args.true_sigma2,
        seed=args.seed, N=args.N_pop, prob=args.prob,
    )
    q_fits_subset = [q_fits_all[i] for i in idx]
    q_true_subset = q_true_all[idx]
    n_oblate = np.sum(q_true_subset < 1.0)
    n_prolate = np.sum(q_true_subset >= 1.0)
    print(f'Selected {len(idx)} streams ({n_oblate} oblate, {n_prolate} prolate)')
    print(f'True q: {np.mean(q_true_subset):.2f} +/- {np.std(q_true_subset):.2f}')

    # Run single population fit (no multiprocessing pool — avoids fd leaks)
    if args.fit_dist == 'gaussian':
        prior_transform = prior_transform_gaussian
    elif args.fit_dist == 'uniform':
        prior_transform = prior_transform_uniform
    elif args.fit_dist == 'binomial':
        prior_transform = prior_transform_binomial

    dns = dynesty.DynamicNestedSampler(
        log_likelihood,
        prior_transform,
        ndim,
        logl_args=(q_fits_subset, args.fit_dist),
        nlive=args.nlive_pop,
        sample='unif',
    )
    dns.run_nested(n_effective=10000)

    res = dns.results
    inds = np.arange(len(res.samples))
    inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]

    mu_med = np.median(samps[:, 0])
    mu_p16, mu_p84 = np.percentile(samps[:, 0], [16, 84])
    print(f'Fitted mu = {mu_med:.3f} ({mu_p16:.3f} -- {mu_p84:.3f})')
    if ndim >= 2:
        sig_med = np.median(samps[:, 1])
        sig_p16, sig_p84 = np.percentile(samps[:, 1], [16, 84])
        print(f'Fitted sigma = {sig_med:.3f} ({sig_p16:.3f} -- {sig_p84:.3f})')

    # Save results
    output_dict = {
        'samps': samps,
        'result': res,
        'q_true_subset': q_true_subset,
        'idx': idx,
        'config': vars(args),
    }
    with open(args.output, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f'Results saved to {args.output}')

    # Corner plot
    if args.fit_dist == 'gaussian':
        fig = corner.corner(samps,
            labels=labels,
            color='blue',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 16})

        axes = np.array(fig.axes).reshape(ndim, ndim)
        # Mark both true means on the mu marginal
        axes[0, 0].axvline(args.true_mu1, color='red', lw=2, ls='--')
        axes[0, 0].axvline(args.true_mu2, color='red', lw=2, ls='--')
        # Mark true sigma on the sigma marginal
        axes[1, 1].axvline(args.true_sigma1, color='red', lw=2, ls='--')
        # Mark on 2D panel too
        axes[1, 0].axvline(args.true_mu1, color='red', lw=1.5, ls='--')
        axes[1, 0].axvline(args.true_mu2, color='red', lw=1.5, ls='--')
        axes[1, 0].axhline(args.true_sigma1, color='red', lw=1.5, ls='--')
    else:
        fig = corner.corner(samps,
            labels=labels,
            color='blue',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 16})

    plot_path = os.path.join(args.path,
        f'corner_mismatch_BtoG_N{args.N_pop}_prob{args.prob}_seed{args.seed}.pdf')
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Corner plot saved to {plot_path}')


if __name__ == '__main__':
    main()
