"""
Cross-check and benchmark our density-flattened NFW against galpy's
TriaxialNFWPotential (the canonical implementation -- C-backed, with a
50-node Gauss-Legendre numerical integral, also flattens the density).

Verifies:
  (1) Potential values agree across a grid of (R, z, q).
  (2) Force values agree.
  (3) Wall-clock speed comparison.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

import astropy.units as u
from galpy.potential import TriaxialNFWPotential, evaluatePotentials, \
                             evaluateRforces, evaluatezforces

from potentials import (
    NFWPotentialFlatDensity, NFWAccelerationFlatDensity,
)
from constants import G

# Our test parameters (physical units: kpc, Msun, Gyr)
logM = 12.0
Rs   = 15.0
M_phys = 10**logM

# halo aligned with z
dirx, diry, dirz = 0.0, 0.0, 1.0


def setup_galpy(q):
    """Build a galpy axisymmetric density-flattened NFW with our parameters."""
    # TriaxialNFW: rho ~ 1/(m(1+m)^2),  m^2 = (x/a)^2 + (y/b a)^2 + (z/c a)^2
    # For axisymmetric (b=1, c=q) -> m^2 = (R^2 + z^2/q^2) / a^2
    # so m_galpy * a = m_ours, exactly the same density.
    #
    # amp = G * M (in galpy internal units with ro=1 kpc, vo=1 km/s).
    pot = TriaxialNFWPotential(
        amp = M_phys * u.Msun,
        a   = Rs * u.kpc,
        b   = 1.0, c = q,
        ro = 8.0, vo = 220.0,   # any choice; we use Quantity inputs anyway
        glorder = 50,            # galpy default for the angular integrals
    )
    return pot


# Convert galpy native units (vo^2) to our units (kpc^2/Gyr^2)
# galpy: Phi is dimensionless in internal units, multiplied by vo^2 (km/s)^2.
# To get into kpc/Gyr scales:
KMS2 = (1 * u.km/u.s)**2
KPC_GYR2 = (1 * u.kpc / u.Gyr)**2
KMS2_TO_KPC2_GYR2 = float((KMS2 / KPC_GYR2).decompose())

KMS = 1 * u.km/u.s
KPCGYR = 1 * u.kpc/u.Gyr
KMS_TO_KPCGYR = float((KMS/KPCGYR).decompose())


# ---------------------------------------------------------------------------
# 1. Accuracy cross-check: Phi values
# ---------------------------------------------------------------------------
print("=" * 70)
print("(1) Phi cross-check: ours vs galpy on a grid of (R, z, q)")
print("=" * 70)

R_vals = [1.0, 5.0, 15.0, 30.0, 60.0]   # kpc
z_vals = [0.5, 5.0, 15.0, 40.0]
q_vals = [0.5, 0.7, 1.0, 1.3, 1.5]

print(f"  Probing {len(R_vals)}x{len(z_vals)} = {len(R_vals)*len(z_vals)} points "
      f"at each of {len(q_vals)} q values.")

for q_ in q_vals:
    pot = setup_galpy(q_)
    max_rel_phi = 0.0
    max_pt = None
    for R_ in R_vals:
        for z_ in z_vals:
            phi_ours = float(NFWPotentialFlatDensity(
                R_, 0.0, z_, logM, Rs, q_, dirx, diry, dirz))
            # galpy returns raw float in (km/s)^2 when use_physical=True
            phi_galpy_kms2 = float(evaluatePotentials(
                pot, R_ * u.kpc, z_ * u.kpc))
            phi_galpy = phi_galpy_kms2 * KMS2_TO_KPC2_GYR2
            rel = abs(phi_ours - phi_galpy) / abs(phi_galpy)
            if rel > max_rel_phi:
                max_rel_phi = rel
                max_pt = (R_, z_, phi_ours, phi_galpy)
    R_, z_, p_o, p_g = max_pt
    print(f"  q = {q_}: max rel err in Phi = {max_rel_phi:.2e}  "
          f"(worst at R={R_}, z={z_}: ours={p_o:.3e}, galpy={p_g:.3e})")


# ---------------------------------------------------------------------------
# 2. Accuracy cross-check: forces
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2) Force cross-check: ours vs galpy")
print("=" * 70)

## galpy returns forces in km/s/Myr when quantity=True; we'll let astropy do
## the conversion to kpc/Gyr^2 below.

for q_ in q_vals:
    pot = setup_galpy(q_)
    max_rel_F = 0.0
    max_pt = None
    for R_ in R_vals:
        for z_ in z_vals:
            # Our analytic acceleration (x = R, y = 0, z = z)
            a_ours = np.asarray(NFWAccelerationFlatDensity(
                R_, 0.0, z_, logM, Rs, q_, dirx, diry, dirz))
            # In Cartesian with y=0, F_R == a_ours[0], F_z == a_ours[2], F_y == 0.
            F_R_ours = a_ours[0]
            F_z_ours = a_ours[2]

            # galpy forces -> astropy Quantity -> kpc/Gyr^2
            F_R_galpy = float(evaluateRforces(pot, R_ * u.kpc, z_ * u.kpc,
                                              quantity=True)
                              .to(u.kpc / u.Gyr**2).value)
            F_z_galpy = float(evaluatezforces(pot, R_ * u.kpc, z_ * u.kpc,
                                              quantity=True)
                              .to(u.kpc / u.Gyr**2).value)

            F_ours  = np.array([F_R_ours,  F_z_ours])
            F_galpy = np.array([F_R_galpy, F_z_galpy])
            rel = np.linalg.norm(F_ours - F_galpy) / np.linalg.norm(F_galpy)
            if rel > max_rel_F:
                max_rel_F = rel
                max_pt = (R_, z_, F_ours, F_galpy)
    R_, z_, Fo, Fg = max_pt
    print(f"  q = {q_}: max rel err in F = {max_rel_F:.2e}")
    print(f"           worst at (R={R_}, z={z_}): "
          f"F_R: ours={Fo[0]:+.3e} galpy={Fg[0]:+.3e}   "
          f"F_z: ours={Fo[1]:+.3e} galpy={Fg[1]:+.3e}")


# ---------------------------------------------------------------------------
# 3. Timing comparison
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(3) Timing: 1000 force evaluations, ours vs galpy")
print("=" * 70)

q_bench = 0.8
pot = setup_galpy(q_bench)

rng = np.random.default_rng(0)
N_pts = 1000
Rs_pts = rng.uniform(1, 80, N_pts)
zs_pts = rng.uniform(-50, 50, N_pts)

# galpy: loop in Python (C kernel per call)
# Conversion factor from native galpy force units to kpc/Gyr^2:
# we compute it once by evaluating a single force in both native and Quantity.
_f_nat = float(evaluateRforces(pot, 10/8.0, 5/8.0, use_physical=False))
_f_phys = float(evaluateRforces(pot, 10*u.kpc, 5*u.kpc, quantity=True)
                .to(u.kpc/u.Gyr**2).value)
NATIVE_TO_KPCGYR2 = _f_phys / _f_nat
print(f"  (native -> kpc/Gyr^2 conversion: x{NATIVE_TO_KPCGYR2:.4f})")

def galpy_force_loop_quantity():
    """One-by-one with Quantity input/output (slowest)."""
    out_R = np.empty(N_pts)
    out_z = np.empty(N_pts)
    for i in range(N_pts):
        out_R[i] = float(evaluateRforces(pot, Rs_pts[i] * u.kpc,
                                         zs_pts[i] * u.kpc, quantity=True)
                          .to(u.kpc/u.Gyr**2).value)
        out_z[i] = float(evaluatezforces(pot, Rs_pts[i] * u.kpc,
                                         zs_pts[i] * u.kpc, quantity=True)
                          .to(u.kpc/u.Gyr**2).value)
    return out_R, out_z

def galpy_force_loop_native():
    """One-by-one in native units (skips Quantity overhead)."""
    Ri = Rs_pts / 8.0
    zi = zs_pts / 8.0
    out_R = np.empty(N_pts)
    out_z = np.empty(N_pts)
    for i in range(N_pts):
        out_R[i] = float(evaluateRforces(pot, Ri[i], zi[i], use_physical=False))
        out_z[i] = float(evaluatezforces(pot, Ri[i], zi[i], use_physical=False))
    return out_R * NATIVE_TO_KPCGYR2, out_z * NATIVE_TO_KPCGYR2

# Ours: jit + vmap (preferred path inside the spray loop)
acc_flat_v = jax.jit(jax.vmap(NFWAccelerationFlatDensity,
                              in_axes=(0, 0, 0, None, None, None,
                                       None, None, None)))
xs = jnp.asarray(Rs_pts)
ys = jnp.zeros_like(xs)
zs = jnp.asarray(zs_pts)

# Warm up JIT
_ = acc_flat_v(xs, ys, zs, logM, Rs, q_bench, dirx, diry, dirz).block_until_ready()

def bench(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for o in out:
                if hasattr(o, 'block_until_ready'):
                    o.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times)

t_galpy_quantity = bench(galpy_force_loop_quantity, n=3)
t_galpy_native   = bench(galpy_force_loop_native, n=3)
t_ours           = bench(lambda: acc_flat_v(xs, ys, zs, logM, Rs, q_bench,
                                            dirx, diry, dirz),
                          n=50)

print(f"  {N_pts} (R, z) points, q = {q_bench}, median wall-clock:")
print(f"    galpy (python loop, Quantity I/O):     {t_galpy_quantity*1000:8.2f} ms"
      f"   ({t_galpy_quantity*1e6/N_pts:.2f} us per call)")
print(f"    galpy (python loop, native units):     {t_galpy_native*1000:8.2f} ms"
      f"   ({t_galpy_native*1e6/N_pts:.2f} us per call)")
print(f"    ours  (jit + vmap):                    {t_ours*1000:8.2f} ms"
      f"   ({t_ours*1e6/N_pts:.2f} us per call)")
print()
print(f"  Speed-up (galpy native / ours):  {t_galpy_native/t_ours:.0f}x faster")
print(f"  Speed-up (galpy quantity / ours):{t_galpy_quantity/t_ours:.0f}x faster")
print()
print("  Note: galpy's TriaxialNFWPotential is a C-backed quadrature too,")
print("  but it loops in Python over points (does not accept array inputs).")
print("  Our jit+vmap fuses the per-point loop with the per-node quadrature.")
