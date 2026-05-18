"""
End-to-end benchmark of one likelihood call (= one full forward model)
with the flattening applied to the potential vs to the density.

Loads the seed 46 mock that's used as the paper's example case, recreates
exactly the same inputs the likelihood function would see, then runs
generate_stream_spray_base from both modules and times each.

Also profiles the dominant sub-stages (satellite integration, Hessian
sweep, stream integration) so we can see where the time difference lives.
"""

import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp

# Two parallel forward models -- same code, different NFW backend
from spray_base          import generate_stream_spray_base as gen_pot
from spray_base_flatdens import generate_stream_spray_base as gen_dens

# And the building blocks for per-stage profiling
import spray_base          as sb_pot
import spray_base_flatdens as sb_dens

from likelihoods import log_likelihood_spray_base
from utils import get_track

SEED_EX = 46
NLIVE   = 2000
SIGMA   = 2

# ---------------------------------------------------------------------------
# 1. Load the exact mock data & parameters used in the paper's example case
# ---------------------------------------------------------------------------
path = f'./MockStreams/seed{SEED_EX}'
with open(os.path.join(path, 'dict_stream.pkl'), 'rb') as f:
    dict_data = pickle.load(f)

# True parameters (same convention as in individual_fits.py)
# stored params layout: [logM, Rs, q, dirx, diry, dirz, logm, rs,
#                        x0, y0, z0, vx0, vy0, vz0, time, alpha]
params_full = np.asarray(dict_data['params'])
print(f'Loaded seed={SEED_EX}, true params shape: {params_full.shape}')

# Reproduce the noise the fit would see
r_sig = dict_data['r_bin'] * SIGMA / 100
rng   = np.random.default_rng(int(SEED_EX))
r_err = rng.normal(0, r_sig)
dict_data['r_bin'] += r_err
dict_data['r_sig']  = r_sig
dict_data['x_bin']  = dict_data['r_bin'] * np.cos(dict_data['theta_bin'])
dict_data['y_bin']  = dict_data['r_bin'] * np.sin(dict_data['theta_bin'])

# Parameter vector for the spray forward model (the 16-element full version)
seed_spray = 2  # matches individual_fits.py convention

# ---------------------------------------------------------------------------
# 2. Warm up JIT once for both backends
# ---------------------------------------------------------------------------
print('Warming up JIT (first call is much slower because of compilation)...')

t0 = time.perf_counter()
out_pot  = gen_pot(params_full, seed_spray)
for arr in out_pot: arr.block_until_ready()
print(f'  potential-flat first call (JIT compile): {time.perf_counter()-t0:6.2f} s')

t0 = time.perf_counter()
out_dens = gen_dens(params_full, seed_spray)
for arr in out_dens: arr.block_until_ready()
print(f'  density-flat   first call (JIT compile): {time.perf_counter()-t0:6.2f} s')


def bench(fn, *args, n=15):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        # block on any returned array
        if isinstance(out, (tuple, list)):
            for o in out:
                if hasattr(o, 'block_until_ready'):
                    o.block_until_ready()
        elif hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.std(times)


# ---------------------------------------------------------------------------
# 3. End-to-end stream generation
# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('End-to-end: generate_stream_spray_base (one full forward model)')
print('=' * 70)

med_pot, std_pot   = bench(gen_pot,  params_full, seed_spray)
med_dens, std_dens = bench(gen_dens, params_full, seed_spray)
print(f'  potential-flat:   {med_pot*1000:7.1f} ms  (sigma {std_pot*1000:.1f})')
print(f'  density-flat:     {med_dens*1000:7.1f} ms  (sigma {std_dens*1000:.1f})')
print(f'  overall slowdown: x{med_dens/med_pot:.2f}')


# ---------------------------------------------------------------------------
# 4. Per-stage breakdown
# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('Per-stage breakdown (where does the difference live?)')
print('=' * 70)

# Unpack params
(logM, Rs, q, dirx, diry, dirz,
 logm, rs,
 x0, y0, z0, vx0, vy0, vz0,
 t_evol, alpha) = params_full

# Convert v from km/s to internal units (mirrors generate_stream_spray_base)
from constants import KMS_TO_KPCGYR
vx0_i = float(vx0) * KMS_TO_KPCGYR
vy0_i = float(vy0) * KMS_TO_KPCGYR
vz0_i = float(vz0) * KMS_TO_KPCGYR

# (a) Satellite integration (200 force calls, scalar inputs)
print()
print('  (a) integrate_satellite (1 backward + 1 forward = 200 force calls each)')
def sat_pot():
    bwd = sb_pot.integrate_satellite(x0, y0, z0, vx0_i, vy0_i, vz0_i,
                                     logM, Rs, q, dirx, diry, dirz, -t_evol)
    fwd = sb_pot.integrate_satellite(*bwd[-1, :],
                                     logM, Rs, q, dirx, diry, dirz, t_evol*alpha)
    return fwd
def sat_dens():
    bwd = sb_dens.integrate_satellite(x0, y0, z0, vx0_i, vy0_i, vz0_i,
                                      logM, Rs, q, dirx, diry, dirz, -t_evol)
    fwd = sb_dens.integrate_satellite(*bwd[-1, :],
                                      logM, Rs, q, dirx, diry, dirz, t_evol*alpha)
    return fwd

_ = sat_pot().block_until_ready()
_ = sat_dens().block_until_ready()
m_pot, _  = bench(sat_pot)
m_dens, _ = bench(sat_dens)
print(f'      potential-flat:  {m_pot*1000:7.2f} ms')
print(f'      density-flat:    {m_dens*1000:7.2f} ms  (x{m_dens/m_pot:.2f})')

# (b) Hessian sweep over forward trajectory (100 Hessian calls, vmapped)
print()
print('  (b) Hessian sweep over satellite trajectory (100 vmapped Hessians)')
fwd_pot  = sat_pot()
fwd_dens = sat_dens()
from potentials import NFWHessian, NFWHessianFlatDensity
hess_pot_v  = jax.jit(jax.vmap(NFWHessian,             in_axes=(0,0,0,None,None,None,None,None,None)))
hess_dens_v = jax.jit(jax.vmap(NFWHessianFlatDensity,  in_axes=(0,0,0,None,None,None,None,None,None)))
xs, ys, zs = fwd_pot[:,0], fwd_pot[:,1], fwd_pot[:,2]
_ = hess_pot_v(xs, ys, zs, logM, Rs, q, dirx, diry, dirz).block_until_ready()
_ = hess_dens_v(xs, ys, zs, logM, Rs, q, dirx, diry, dirz).block_until_ready()
m_pot, _  = bench(hess_pot_v, xs, ys, zs, logM, Rs, q, dirx, diry, dirz)
m_dens, _ = bench(hess_dens_v, xs, ys, zs, logM, Rs, q, dirx, diry, dirz)
print(f'      potential-flat:  {m_pot*1000:7.2f} ms')
print(f'      density-flat:    {m_dens*1000:7.2f} ms  (x{m_dens/m_pot:.2f})')

# (c) Stream integration -- the dominant cost. The simplest way to measure
# is overall minus (a) minus (b).
print()
print('  (c) Stream integration (10000 particles x 100 steps x 4 force calls)')
print(f'      potential-flat:  {(med_pot - m_pot - 0)*1000:7.2f} ms (approximate)')
print(f'      density-flat:    {(med_dens - m_dens - 0)*1000:7.2f} ms (approximate)')


# ---------------------------------------------------------------------------
# 5. Full log-likelihood call (forward model + track extraction + chi^2)
# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('Full log-likelihood call (model + binning + chi^2)')
print('=' * 70)

# The fit parameter vector is the 13-element reduced one used by Dynesty.
# layout: [logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, t0]
params_fit = np.concatenate([params_full[:2], params_full[3:9],
                             params_full[10:-1]])

# The likelihood function in likelihoods.py imports spray_base directly,
# so we have to monkey-patch it for the density-flat run.
import likelihoods as lk
orig_gen = lk.generate_stream_spray_base

def loglike_with(backend, n=10):
    lk.generate_stream_spray_base = backend
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        v = lk.log_likelihood_spray_base(params_fit, dict_data, seed=seed_spray)
        times.append(time.perf_counter() - t0)
    return np.median(times), v

# warm up
_ = lk.log_likelihood_spray_base(params_fit, dict_data, seed=seed_spray)
m_pot,  v_pot  = loglike_with(orig_gen)
m_dens, v_dens = loglike_with(gen_dens)
lk.generate_stream_spray_base = orig_gen

print(f'  potential-flat:  {m_pot*1000:7.1f} ms   logL = {v_pot:.3f}')
print(f'  density-flat:    {m_dens*1000:7.1f} ms   logL = {v_dens:.3f}')
print(f'  slowdown: x{m_dens/m_pot:.2f}')

print()
print('Note: at the true parameters, both models should produce a stream of')
print('similar shape but different physics, so logL values will differ.')
print('What matters here is wall-clock time per likelihood evaluation.')
