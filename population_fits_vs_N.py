"""
Population fit convergence vs N — for referee Comment 14.

For each N in a list of stream counts, runs n_bootstrap population fits
(each with a different random subset of streams) and records:
  - bias on mu_pop:  |median(mu_posterior) - true_mu|
  - width on mu_pop: p84 - p16 of mu_posterior

Results are saved to a pickle for plotting.

Usage:
    python population_fits_vs_N.py \
        --path ./MockStreams \
        --true_dist gaussian --true_mu 1.0 --true_sigma 0.1 \
        --fit_dist gaussian \
        --N_list 5 10 15 20 25 35 50 \
        --n_bootstrap 10 \
        --nlive_ind 2000 --sigma_noise 2 \
        --nlive_pop 500
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from population_fits import (
    dynesty_fit,
    subset_as_gaussian,
    subset_as_uniform,
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


def select_subset(q_true, true_dist, true_mu, true_sigma, seed, N):
    """Select a subset of N streams matching the true population distribution."""
    if true_dist == 'gaussian':
        return subset_as_gaussian(q_true, true_mu, true_sigma, seed=seed, N=N)
    elif true_dist == 'uniform':
        return subset_as_uniform(q_true, true_mu, true_sigma, seed=seed, N=N)
    elif true_dist == 'binomial':
        return subset_as_binomial(q_true, true_mu, true_sigma, seed=seed, N=N)
    else:
        raise ValueError(f'Unknown distribution: {true_dist}')


def run_one_fit(q_fits_subset, fit_dist, nlive_pop):
    """Run a single population fit and return posterior samples."""
    if fit_dist == 'gaussian':
        ndim = 2
    elif fit_dist == 'uniform':
        ndim = 2
    elif fit_dist == 'binomial':
        ndim = 5
    else:
        raise ValueError(f'Unknown fit distribution: {fit_dist}')

    result = dynesty_fit(q_fits_subset, ndim=ndim, nlive=nlive_pop, pop_type=fit_dist)
    return result['samps']


def main():
    parser = argparse.ArgumentParser(description='Population fit convergence vs N')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to MockStreams directory')
    parser.add_argument('--true_dist', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'binomial'],
                        help='True underlying population distribution')
    parser.add_argument('--true_mu', type=float, default=1.0,
                        help='True population mean (or centre for uniform)')
    parser.add_argument('--true_sigma', type=float, default=0.1,
                        help='True population sigma (or half-width for uniform)')
    parser.add_argument('--fit_dist', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'binomial'],
                        help='Population model to fit')
    parser.add_argument('--N_list', type=int, nargs='+', default=[5, 10, 15, 20, 25, 35, 50],
                        help='List of N values to test')
    parser.add_argument('--n_bootstrap', type=int, default=10,
                        help='Number of bootstrap iterations per N')
    parser.add_argument('--nlive_ind', type=int, default=2000,
                        help='nlive used in individual fits (for loading)')
    parser.add_argument('--sigma_noise', type=int, default=2,
                        help='Sigma label used in individual fits (for loading)')
    parser.add_argument('--nlive_pop', type=int, default=500,
                        help='nlive for population fits')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle path (default: <path>/pop_vs_N_results.pkl)')
    args = parser.parse_args()

    if args.output is None:
        N_tag = '_'.join(str(n) for n in args.N_list)
        args.output = os.path.join(args.path, f'pop_vs_N_{N_tag}_results.pkl')

    # Load all individual fits once
    q_true_all, q_fits_all = load_all_streams(args.path, args.nlive_ind, args.sigma_noise)
    n_available = len(q_true_all)

    results = {}

    for N in args.N_list:
        if N > n_available:
            print(f'Skipping N={N}: only {n_available} streams available')
            continue

        print(f'\n=== N = {N} ({args.n_bootstrap} bootstrap runs) ===')
        bias_list = []
        width_mu_list = []
        width_sigma_list = []

        for b in range(args.n_bootstrap):
            boot_seed = 1000 * N + b  # unique seed per (N, bootstrap)
            print(f'  Bootstrap {b+1}/{args.n_bootstrap} (seed={boot_seed})')

            # Select subset
            try:
                idx = select_subset(q_true_all, args.true_dist,
                                    args.true_mu, args.true_sigma,
                                    seed=boot_seed, N=N)
            except Exception as e:
                print(f'    Subset selection failed: {e}, skipping')
                continue

            q_fits_subset = [q_fits_all[i] for i in idx]

            # Run population fit
            samps = run_one_fit(q_fits_subset, args.fit_dist, args.nlive_pop)

            # mu is the first parameter for gaussian/uniform
            mu_samps = samps[:, 0]
            mu_median = np.median(mu_samps)
            mu_p16, mu_p84 = np.percentile(mu_samps, [16, 84])

            bias = np.abs(mu_median - args.true_mu)
            width_mu = mu_p84 - mu_p16

            # sigma is the second parameter for gaussian
            if args.fit_dist == 'gaussian':
                sig_samps = samps[:, 1]
                sig_p16, sig_p84 = np.percentile(sig_samps, [16, 84])
                width_sigma = sig_p84 - sig_p16
            else:
                width_sigma = np.nan

            bias_list.append(bias)
            width_mu_list.append(width_mu)
            width_sigma_list.append(width_sigma)

            print(f'    mu_median={mu_median:.3f}, bias={bias:.3f}, '
                  f'width_mu={width_mu:.3f}, width_sigma={width_sigma:.3f}')

        results[N] = {
            'bias': np.array(bias_list),
            'width_mu': np.array(width_mu_list),
            'width_sigma': np.array(width_sigma_list),
            'bias_mean': np.mean(bias_list),
            'bias_std': np.std(bias_list),
            'width_mu_mean': np.mean(width_mu_list),
            'width_mu_std': np.std(width_mu_list),
            'width_sigma_mean': np.mean(width_sigma_list),
            'width_sigma_std': np.std(width_sigma_list),
        }

        print(f'  => bias = {results[N]["bias_mean"]:.3f} +/- {results[N]["bias_std"]:.3f}')
        print(f'  => width_mu = {results[N]["width_mu_mean"]:.3f} +/- {results[N]["width_mu_std"]:.3f}')
        print(f'  => width_sigma = {results[N]["width_sigma_mean"]:.3f} +/- {results[N]["width_sigma_std"]:.3f}')

    # Save
    output_dict = {
        'N_list': [N for N in args.N_list if N <= n_available],
        'results': results,
        'config': vars(args),
    }
    with open(args.output, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
