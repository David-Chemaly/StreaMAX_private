"""
Tests and benchmarks for NFWPotentialFlatDensity.

Three checks:
  (1) At q = 1, the flat-density NFW must reproduce the closed-form spherical
      NFW potential AND its gradient to ~quadrature precision.
  (2) For arbitrary q != 1, recovering rho via Poisson (rho = Lap Phi / 4 pi G)
      from the autodiff Hessian must equal the analytic rho_NFW(m) used to
      construct the density. This is the integrity test.
  (3) Single-call and batched (vmap) timing vs the existing closed-form
      NFWPotential / NFWAcceleration.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from potentials import (
    NFWPotential, NFWAcceleration, NFWHessian,
    NFWPotentialFlatDensity, NFWAccelerationFlatDensity, NFWHessianFlatDensity,
    NFWdHessianFlatDensity,
)
from constants import G


# Common test parameters
logM = 12.0
Rs   = 15.0   # kpc
dirx, diry, dirz = 0.0, 0.0, 1.0  # halo axis aligned with z


# ---------------------------------------------------------------------------
# (1) q = 1 sanity check: must equal the closed-form analytic spherical NFW
# ---------------------------------------------------------------------------
def analytic_spherical_nfw_phi(r):
    M = 10**logM
    return -G * M * np.log(1.0 + r / Rs) / r

def analytic_spherical_nfw_force(r):
    """Magnitude of radial force (-d Phi / d r)."""
    M = 10**logM
    return -G * M * (1.0 / (r * (Rs + r)) - np.log(1.0 + r / Rs) / r**2)

print("=" * 70)
print("(1) q = 1 limit: flat-density NFW vs closed-form spherical NFW")
print("=" * 70)

# Scan over a range of radii from 0.1 Rs to 10 Rs, multiple directions
rng = np.random.default_rng(0)
N_test = 200
rs = np.geomspace(0.5, 150.0, N_test)
# random direction unit vectors
dirs = rng.normal(size=(N_test, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
positions = rs[:, None] * dirs

phi_analytic = analytic_spherical_nfw_phi(rs)
phi_flat = np.array([
    float(NFWPotentialFlatDensity(float(p[0]), float(p[1]), float(p[2]),
                                  logM, Rs, 1.0, dirx, diry, dirz))
    for p in positions
])
# also the potential-flatten q=1 version (which IS the closed form)
phi_pot = np.array([
    float(NFWPotential(float(p[0]), float(p[1]), float(p[2]),
                       logM, Rs, 1.0, dirx, diry, dirz))
    for p in positions
])

rel_flat = np.abs((phi_flat - phi_analytic) / phi_analytic)
rel_pot  = np.abs((phi_pot  - phi_analytic) / phi_analytic)
print(f"  Phi:   flat-density max rel err vs analytic = {rel_flat.max():.2e}")
print(f"         potential-flat   max rel err vs analytic = {rel_pot.max():.2e}")

# Gradients should match too
acc_pot_q1 = np.array([
    np.asarray(NFWAcceleration(float(p[0]), float(p[1]), float(p[2]),
                               logM, Rs, 1.0, dirx, diry, dirz))
    for p in positions
])
acc_flat_q1 = np.array([
    np.asarray(NFWAccelerationFlatDensity(float(p[0]), float(p[1]), float(p[2]),
                                          logM, Rs, 1.0, dirx, diry, dirz))
    for p in positions
])
acc_diff = np.linalg.norm(acc_flat_q1 - acc_pot_q1, axis=1) / np.linalg.norm(acc_pot_q1, axis=1)
print(f"  Acc:   max rel ||acc_flat - acc_pot|| / ||acc_pot|| at q=1 = {acc_diff.max():.2e}")


# ---------------------------------------------------------------------------
# (2) Poisson recovery test: does Lap Phi / (4 pi G) equal the analytic rho?
# ---------------------------------------------------------------------------
def analytic_rho_nfw(R, z, q):
    """Density used to define the flat-density model: rho_NFW(m), m^2 = R^2 + z^2/q^2."""
    M = 10**logM
    rho_s = M / (4.0 * np.pi * Rs**3)
    m = np.sqrt(R**2 + (z / q)**2)
    s = m / Rs
    return rho_s / (s * (1.0 + s)**2)

print()
print("=" * 70)
print("(2) Poisson recovery: rho from Lap(Phi)/(4 pi G) vs analytic rho_NFW")
print("=" * 70)

for q in [0.5, 0.7, 1.0, 1.3, 1.5]:
    test_pts = [
        (5.0,  0.0,  0.1),   # near equator
        (5.0,  0.0,  5.0),   # mid
        (10.0, 0.0, 20.0),   # off-axis
        (0.1,  0.0, 10.0),   # near pole
        (30.0, 0.0, 30.0),   # outer
    ]
    print(f"  q = {q}")
    max_err = 0.0
    for (Rt, yt, zt) in test_pts:
        H = NFWHessianFlatDensity(float(Rt), float(yt), float(zt),
                                  logM, Rs, q, dirx, diry, dirz)
        lap = float(H[0, 0] + H[1, 1] + H[2, 2])
        rho_num = lap / (4.0 * np.pi * G)
        rho_ana = analytic_rho_nfw(Rt, zt, q)
        rel = abs(rho_num - rho_ana) / rho_ana
        max_err = max(max_err, rel)
        print(f"    pt=({Rt:5.1f},{zt:5.1f}): rho_num={rho_num:.4e}  rho_ana={rho_ana:.4e}  rel_err={rel:.2e}")
    print(f"    max rel err for q={q}: {max_err:.2e}\n")


# ---------------------------------------------------------------------------
# (2b) Analytic vs autodiff acceleration at q != 1
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2b) Analytic acceleration vs jax.grad(Phi) at q != 1")
print("=" * 70)

def autodiff_acc(x, y, z, q):
    f = lambda p: NFWPotentialFlatDensity(p[0], p[1], p[2],
                                          logM, Rs, q, dirx, diry, dirz)
    return -jax.grad(f)(jnp.array([x, y, z]))

for q in [0.5, 0.7, 1.0, 1.3, 1.5]:
    test_pts = [(5., 0., 0.1), (5., 0., 5.), (10., 0., 20.),
                (0.1, 0., 10.), (30., 0., 30.)]
    max_rel = 0.0
    for (xt, yt, zt) in test_pts:
        a_an = np.asarray(NFWAccelerationFlatDensity(xt, yt, zt,
                          logM, Rs, q, dirx, diry, dirz))
        a_ad = np.asarray(autodiff_acc(xt, yt, zt, q))
        rel  = np.linalg.norm(a_an - a_ad) / np.linalg.norm(a_ad)
        max_rel = max(max_rel, rel)
    print(f"  q={q}: max rel ||a_analytic - a_autodiff|| / ||a|| = {max_rel:.2e}")


# ---------------------------------------------------------------------------
# (3) Timing benchmark
# ---------------------------------------------------------------------------
print("=" * 70)
print("(3) Timing benchmark")
print("=" * 70)

# Batched evaluation via vmap
xs = jnp.asarray(positions[:, 0])
ys = jnp.asarray(positions[:, 1])
zs = jnp.asarray(positions[:, 2])

# Vectorise over position only
phi_pot_v   = jax.jit(jax.vmap(NFWPotential,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))
phi_flat_v  = jax.jit(jax.vmap(NFWPotentialFlatDensity,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))
acc_pot_v   = jax.jit(jax.vmap(NFWAcceleration,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))
acc_flat_v  = jax.jit(jax.vmap(NFWAccelerationFlatDensity,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))

q_bench = 0.8

# Warm-up JIT
_ = phi_pot_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()
_ = phi_flat_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()
_ = acc_pot_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()
_ = acc_flat_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()

def bench(fn, *args, n=200):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.std(times)

print(f"  Batched over N={N_test} positions, q={q_bench}, median of 200 trials:")
m, s = bench(phi_pot_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Phi (potential-flat ): {m*1e6:8.2f} us")
m_flat, _ = bench(phi_flat_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Phi (density-flat   ): {m_flat*1e6:8.2f} us   (x{m_flat/m:.1f} slower)")
m, _ = bench(acc_pot_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Acc (potential-flat ): {m*1e6:8.2f} us")
m_flat, _ = bench(acc_flat_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Acc (density-flat   ): {m_flat*1e6:8.2f} us   (x{m_flat/m:.1f} slower)")

hess_pot_v  = jax.jit(jax.vmap(NFWHessian,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))
hess_flat_v = jax.jit(jax.vmap(NFWHessianFlatDensity,
                                in_axes=(0, 0, 0, None, None, None, None, None, None)))
_ = hess_pot_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()
_ = hess_flat_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()
m, _ = bench(hess_pot_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Hess (potential-flat): {m*1e6:8.2f} us")
m_flat, _ = bench(hess_flat_v, xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz)
print(f"    Hess (density-flat  ): {m_flat*1e6:8.2f} us   (x{m_flat/m:.1f} slower)")

print()
print("Done.")
