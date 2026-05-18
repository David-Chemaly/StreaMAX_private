"""
Thorough correctness tests for NFWPotentialFlatDensity.

Each test stresses a different aspect of the implementation; together they
provide strong evidence that the potential and its derivatives are
self-consistent and physically correct.

Tests:
  (A) Quadrature convergence: Phi(N) -> Phi(infty) at fixed point as N grows.
  (B) Dense Poisson recovery: 2D heatmap of rel err in rho across (R, z)
      for q in {0.5, 0.7, 1.0, 1.3, 1.5}. The single strongest correctness
      test we have -- it confirms the potential's Laplacian reproduces the
      analytic flattened-density NFW everywhere, not just at a few points.
  (C) Symmetries:
        - axisymmetry (Phi independent of azimuth phi)
        - reflection z -> -z
        - force has zero phi component
        - on the z-axis the force is purely along z
        - on the equator the force is purely radial in (x, y)
        - at q=1 the potential at fixed |r| is independent of halo orientation
  (D) Energy and Lz conservation along a closed orbit integrated with
      symplectic leapfrog. Forces are wrong if E drifts.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

from potentials import (
    NFWPotentialFlatDensity,
    NFWAccelerationFlatDensity,
    NFWHessianFlatDensity,
)
from constants import G

# Common test parameters
logM = 12.0
Rs   = 15.0
M_phys = 10**logM


# ---------------------------------------------------------------------------
# Helper: build a flat-density NFW potential with a SETTABLE N_QUAD.
# This is used for the convergence study; the rest of the tests use the
# module's default N (currently 16).
# ---------------------------------------------------------------------------
def make_phi_with_N(N):
    x_gl, w_gl = np.polynomial.legendre.leggauss(N)
    V = jnp.asarray(0.5 * (x_gl + 1.0))
    W = jnp.asarray(0.5 * w_gl)
    EPS = 1e-8

    @jax.jit
    def phi(R, Z, q):
        xi   = R / Rs
        zeta = Z / Rs
        v   = V
        q2  = q * q
        D   = q2 + (1.0 - q2) * v * v
        m   = v * jnp.sqrt(xi*xi/D + zeta*zeta/q2 + EPS)
        integrand = 1.0 / ((1.0 + m) * D)
        I = jnp.sum(W * integrand)
        return -(G * M_phys * q2 / Rs) * I
    return phi


# ---------------------------------------------------------------------------
# (A) Quadrature convergence study
# ---------------------------------------------------------------------------
print("=" * 70)
print("(A) Quadrature convergence: Phi vs N at several test points")
print("=" * 70)

Ns = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
test_pts = [(1.0, 0.0, 0.0), (5.0, 0.0, 5.0), (15.0, 0.0, 30.0)]
test_qs  = [0.5, 1.0, 1.5]
phi_inf = {}
for (R_, _, Z_) in test_pts:
    for q_ in test_qs:
        phi_inf[(R_, Z_, q_)] = float(make_phi_with_N(256)(R_, Z_, q_))

print(f"  {'N':>4} | " + " | ".join(f"q={q}, R={R:.0f}, z={Z:.0f}"
                                     for (R,_,Z) in test_pts for q in test_qs))
for N in Ns:
    phi = make_phi_with_N(N)
    row = [f"{N:4d}"]
    for (R_, _, Z_) in test_pts:
        for q_ in test_qs:
            ref = phi_inf[(R_, Z_, q_)]
            val = float(phi(R_, Z_, q_))
            rel = abs(val - ref) / abs(ref)
            row.append(f"{rel:.1e}")
    print("  " + " | ".join(row))
print("  (Values are |Phi(N) - Phi(N=256)| / |Phi(N=256)|;")
print("   we want this to fall geometrically with N and saturate near float32 ~1e-7.)")


# ---------------------------------------------------------------------------
# (B) Dense Poisson recovery heatmaps
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(B) Dense Poisson recovery: rho from Lap(Phi)/(4 pi G) vs analytic")
print("=" * 70)

dirx, diry, dirz = 0.0, 0.0, 1.0

def analytic_rho_nfw(R, z, q):
    rho_s = M_phys / (4.0 * np.pi * Rs**3)
    m = np.sqrt(R**2 + (z / q)**2)
    s = m / Rs
    return rho_s / (s * (1.0 + s)**2)

def rho_numerical(R, z, q):
    H = NFWHessianFlatDensity(float(R), 0.0, float(z),
                              logM, Rs, q, dirx, diry, dirz)
    return float(H[0, 0] + H[1, 1] + H[2, 2]) / (4.0 * np.pi * G)

N_grid = 41
R_grid = np.geomspace(0.5, 5 * Rs, N_grid)  # avoid R=0 singularity in rho
z_grid = np.geomspace(0.5, 5 * Rs, N_grid)
RR, ZZ = np.meshgrid(R_grid, z_grid, indexing='xy')

fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
max_rel_per_q = {}
for k, q_ in enumerate([0.5, 0.7, 1.0, 1.3, 1.5]):
    rel = np.zeros_like(RR)
    for i in range(N_grid):
        for j in range(N_grid):
            rn = rho_numerical(RR[i, j], ZZ[i, j], q_)
            ra = analytic_rho_nfw(RR[i, j], ZZ[i, j], q_)
            rel[i, j] = abs(rn - ra) / abs(ra)
    max_rel_per_q[q_] = rel.max()
    print(f"  q={q_}: max rel err on {N_grid}x{N_grid} grid = {rel.max():.2e}, "
          f"median = {np.median(rel):.2e}")
    im = axes[k].pcolormesh(RR / Rs, ZZ / Rs, np.log10(rel + 1e-30),
                              cmap='viridis', vmin=-9, vmax=-4, shading='auto')
    axes[k].set_title(f'$q = {q_}$\nmax rel err = {rel.max():.1e}')
    axes[k].set_xlabel(r'$R / R_s$')
    if k == 0:
        axes[k].set_ylabel(r'$z / R_s$')
    axes[k].set_aspect('equal')
fig.colorbar(im, ax=axes[-1], label=r'$\log_{10}|\rho_\text{num} - \rho_\text{ana}|/\rho_\text{ana}$')
fig.suptitle('Dense Poisson recovery test: relative error across (R, z)')
out_pdf = './Plots/flatdens_poisson_recovery.pdf'
plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
print(f"  Heatmap saved -> {out_pdf}")
plt.close()


# ---------------------------------------------------------------------------
# (C) Symmetry tests
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(C) Symmetry tests")
print("=" * 70)

def Phi(x, y, z, q, dxs=(0., 0., 1.)):
    return float(NFWPotentialFlatDensity(float(x), float(y), float(z),
                                         logM, Rs, q, *dxs))

def Acc(x, y, z, q, dxs=(0., 0., 1.)):
    return np.asarray(NFWAccelerationFlatDensity(float(x), float(y), float(z),
                                                 logM, Rs, q, *dxs))

# C1: Axisymmetry -- Phi(R cos phi, R sin phi, z) independent of phi
print("  (C1) Axisymmetry (Phi at fixed (R, z), 20 azimuthal angles)")
worst = 0.0
for (R_, z_) in [(2., 1.), (10., -5.), (20., 30.)]:
    for q_ in [0.5, 1.0, 1.5]:
        vals = [Phi(R_ * np.cos(p), R_ * np.sin(p), z_, q_)
                for p in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
        spread = (max(vals) - min(vals)) / abs(np.mean(vals))
        worst = max(worst, spread)
print(f"      worst relative spread over phi: {worst:.2e}")

# C2: Reflection z -> -z
print("  (C2) Mirror symmetry under z -> -z")
worst = 0.0
for (R_, z_) in [(2., 5.), (10., 20.), (1., 50.)]:
    for q_ in [0.5, 1.0, 1.5]:
        a = Phi(R_, 0., z_, q_)
        b = Phi(R_, 0., -z_, q_)
        worst = max(worst, abs(a - b) / abs(a))
print(f"      worst |Phi(R,z) - Phi(R,-z)| / |Phi|: {worst:.2e}")

# C3: Force azimuthal component should be zero
print("  (C3) F_phi == 0 (force has no azimuthal component)")
worst = 0.0
for (R_, z_) in [(2., 1.), (10., -5.), (20., 30.)]:
    for q_ in [0.5, 1.0, 1.5]:
        for phi_ in np.linspace(0.1, 2 * np.pi - 0.1, 7):
            x_, y_ = R_ * np.cos(phi_), R_ * np.sin(phi_)
            ax, ay, az = Acc(x_, y_, z_, q_)
            # phi-hat = (-sin phi, cos phi, 0)
            F_phi = -np.sin(phi_) * ax + np.cos(phi_) * ay
            F_mag = np.linalg.norm([ax, ay, az])
            worst = max(worst, abs(F_phi) / F_mag)
print(f"      worst |F_phi| / |F|: {worst:.2e}")

# C4: On z-axis (R=0), force has no R component (purely along z).
# Test at exactly R=0 -- EPSILON inside the code handles the sqrt cleanly,
# and rx/R * rx = 0 exactly when rx = 0.
# Also test the scaling F_R ~ R for small R, since that's pure physics.
print("  (C4) F_xy == 0 on the z-axis (exact R=0) and scaling F_R ~ R")
worst_exact = 0.0
for z_ in [1., 5., 20., -10.]:
    for q_ in [0.5, 1.0, 1.5]:
        ax, ay, az = Acc(0.0, 0.0, z_, q_)
        F_R = np.sqrt(ax**2 + ay**2)
        worst_exact = max(worst_exact, F_R / abs(az))
print(f"      worst F_R / |F_z| at exact R=0: {worst_exact:.2e}")

# Verify F_R scales linearly with R near the axis (Taylor expansion of
# axisymmetric force has F_R = R * Phi_RR(0, z) + O(R^3))
slope_ratios = []
for z_ in [1., 5., 20.]:
    for q_ in [0.5, 1.0, 1.5]:
        F_R_per_R = []
        for R_ in [1e-2, 1e-3, 1e-4]:
            ax, ay, az = Acc(R_, 0.0, z_, q_)
            F_R = np.sqrt(ax**2 + ay**2)
            F_R_per_R.append(F_R / R_)
        # F_R/R should be constant across the three R values
        slope_ratios.append(max(F_R_per_R) / min(F_R_per_R) - 1.0)
print(f"      (sanity) max relative drift of F_R/R as R shrinks: {max(slope_ratios):.2e}")
print(f"               (a flat F_R/R confirms F_R ~ R physics, not numerical noise)")

# C5: On equator (z=0), force has no z component (purely radial)
print("  (C5) F_z == 0 on the equator")
worst = 0.0
for R_ in [1., 5., 20., 50.]:
    for q_ in [0.5, 1.0, 1.5]:
        ax, ay, az = Acc(R_, 0.0, 0.0, q_)
        F_R = np.sqrt(ax**2 + ay**2)
        worst = max(worst, abs(az) / F_R)
print(f"      worst |F_z| / |F_R| on equator: {worst:.2e}")

# C6: At q=1 the potential should be invariant under arbitrary halo rotation
print("  (C6) q=1 rotation invariance (different halo axes, same |r|)")
worst = 0.0
for r_ in [3., 15., 60.]:
    # 6 different points on a sphere of radius r_
    pts = [(r_, 0, 0), (0, r_, 0), (0, 0, r_),
           (r_/np.sqrt(2), r_/np.sqrt(2), 0),
           (r_/np.sqrt(3),)*3,
           (-r_/np.sqrt(2), 0, r_/np.sqrt(2))]
    for axis in [(0,0,1), (1,0,0), (0,1,0), (1,1,0), (1,1,1), (-1,2,3)]:
        ax = np.asarray(axis, dtype=float); ax /= np.linalg.norm(ax)
        vals = [Phi(*p, 1.0, dxs=tuple(ax)) for p in pts]
        spread = (max(vals) - min(vals)) / abs(np.mean(vals))
        worst = max(worst, spread)
print(f"      worst spread over (point, halo axis): {worst:.2e}")


# ---------------------------------------------------------------------------
# (D) Energy and L_z conservation along an orbit (kick-drift-kick leapfrog)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(D) Energy and L_z conservation along a 100-period orbit")
print("=" * 70)

def step_leapfrog(state, dt, q):
    x, y, z, vx, vy, vz = state
    ax, ay, az = Acc(x, y, z, q)
    vx += 0.5 * dt * ax
    vy += 0.5 * dt * ay
    vz += 0.5 * dt * az
    x  += dt * vx; y += dt * vy; z += dt * vz
    ax, ay, az = Acc(x, y, z, q)
    vx += 0.5 * dt * ax
    vy += 0.5 * dt * ay
    vz += 0.5 * dt * az
    return np.array([x, y, z, vx, vy, vz])

# Initial conditions: roughly circular orbit at R0 in the equatorial plane,
# but with a small z velocity so it explores the z structure (only matters
# for q != 1).
for q_ in [0.5, 1.0, 1.5]:
    R0 = 20.0
    # v_circ from |F_R| * R = v^2 (for q=1; near-circular for q!=1)
    ax, ay, az = Acc(R0, 0.0, 0.0, q_)
    v_circ = np.sqrt(-ax * R0)
    state = np.array([R0, 0.0, 0.0, 0.0, v_circ, 0.1 * v_circ])

    # Pick a dt small enough that leapfrog conserves E to ~1e-6 over 100 orbits
    T_orb = 2 * np.pi * R0 / v_circ
    n_steps_per_orbit = 200
    n_orbits = 100
    dt = T_orb / n_steps_per_orbit
    n_steps = n_steps_per_orbit * n_orbits

    Es, Lzs, ts = [], [], []
    def energy(s, q):
        x, y, z, vx, vy, vz = s
        T = 0.5 * (vx**2 + vy**2 + vz**2)
        U = Phi(x, y, z, q)
        return T + U
    def Lz(s):
        x, y, z, vx, vy, vz = s
        return x * vy - y * vx

    Es.append(energy(state, q_));  Lzs.append(Lz(state));  ts.append(0.0)
    for n in range(n_steps):
        state = step_leapfrog(state, dt, q_)
        if (n + 1) % 100 == 0:
            Es.append(energy(state, q_));  Lzs.append(Lz(state));
            ts.append((n + 1) * dt)

    Es  = np.array(Es)
    Lzs = np.array(Lzs)
    dE   = np.max(np.abs(Es  - Es[0]))  / abs(Es[0])
    dLz  = np.max(np.abs(Lzs - Lzs[0])) / abs(Lzs[0])
    print(f"  q={q_}: after {n_orbits} orbits, dt = T_orb / {n_steps_per_orbit}")
    print(f"      max |dE/E|  = {dE:.2e}    max |dLz/Lz| = {dLz:.2e}")
    print(f"      (a symplectic leapfrog with correct forces should give"
          f" dE ~ dt^2 ~ {(1/n_steps_per_orbit)**2:.1e})")

print()
print("Done.")
