"""
One-off check: where does the *implied density* of the flattened-potential
NFW (the model in potentials.py) go negative for q = 0.5?

The potential is:
    Phi(R, z) = -G M / r * ln(1 + r / Rs),    r^2 = R^2 + (z/q)^2
Density is recovered via Poisson:
    rho = (1/4 pi G) Laplacian(Phi)
For q != 1 this can be negative.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from potentials import NFWPotential
from constants import G

# Typical NFW for a Milky-Way-mass host, axis aligned with z
logM = 12.0
Rs   = 15.0     # kpc
q    = 0.7
dirx, diry, dirz = 0.0, 0.0, 1.0  # halo z-axis along world z

# Axisymmetric: y = 0, scan in (R, z). Use abs(z) by symmetry.
N = 401
R_grid = np.linspace(0.01, 5 * Rs, N)  # kpc
z_grid = np.linspace(0.01, 5 * Rs, N)
RR, ZZ = np.meshgrid(R_grid, z_grid, indexing='xy')

# Build Laplacian via jax.hessian of Phi at each point
def phi_at(x, y, z):
    return NFWPotential(x, y, z, logM, Rs, q, dirx, diry, dirz)

hess_fn = jax.jit(jax.hessian(lambda xyz: phi_at(xyz[0], xyz[1], xyz[2])))

def laplacian(R, z):
    h = hess_fn(jnp.array([R, 0.0, z]))
    return float(h[0, 0] + h[1, 1] + h[2, 2])

lap = np.zeros_like(RR)
for i in range(N):
    for j in range(N):
        lap[i, j] = laplacian(RR[i, j], ZZ[i, j])

rho = lap / (4 * np.pi * G)     # Msun / kpc^3

# Where is rho negative?
neg_mask = rho < 0

print(f'q = {q},  logM = {logM},  Rs = {Rs} kpc')
print(f'Grid covers R, z in [0, {5*Rs:.1f}] kpc, {N}x{N}')
print(f'Fraction of grid with rho<0: {neg_mask.mean()*100:.1f}%')

# Find the boundary by walking outward at fixed z (axis) and at fixed R (equator)
def first_negative_along(axis):
    if axis == 'z':
        # R=0, scan z
        coords = z_grid
        rho_line = np.array([laplacian(0.0, zi) / (4 * np.pi * G) for zi in z_grid])
    elif axis == 'R':
        coords = R_grid
        rho_line = np.array([laplacian(Ri, 0.0) / (4 * np.pi * G) for Ri in R_grid])
    neg_idx = np.where(rho_line < 0)[0]
    if len(neg_idx):
        return coords[neg_idx[0]], coords[neg_idx[-1]], rho_line
    return None, None, rho_line

z_first, z_last, rho_zaxis = first_negative_along('z')
R_first, R_last, rho_equat = first_negative_along('R')

print(f'\nAlong the z-axis (R=0):')
if z_first is None:
    print('  rho >= 0 everywhere on this segment.')
else:
    print(f'  rho<0 from z = {z_first:.2f} kpc  to  z = {z_last:.2f} kpc')
print(f'\nAlong the equator (z=0):')
if R_first is None:
    print('  rho >= 0 everywhere on this segment.')
else:
    print(f'  rho<0 from R = {R_first:.2f} kpc  to  R = {R_last:.2f} kpc')

# Plot: log|rho| with sign, plus rho<0 region outlined
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Left: signed log
sign = np.sign(rho)
log_abs = np.log10(np.abs(rho) + 1e-20)
im = ax[0].pcolormesh(RR / Rs, ZZ / Rs, sign * log_abs,
                       cmap='RdBu_r',
                       vmin=-np.nanmax(log_abs), vmax=np.nanmax(log_abs),
                       shading='auto')
ax[0].contour(RR / Rs, ZZ / Rs, rho, levels=[0.0],
              colors='k', linewidths=2)
ax[0].set_xlabel(r'$R / R_s$')
ax[0].set_ylabel(r'$z / R_s$')
ax[0].set_title(rf'sign($\rho$) $\times \log_{{10}}|\rho|$, $q={q}$')
plt.colorbar(im, ax=ax[0], label=r'sign($\rho$) $\log_{10}|\rho|$ [M$_\odot$/kpc$^3$]')

# Right: just the negative region as a mask
ax[1].pcolormesh(RR / Rs, ZZ / Rs, neg_mask.astype(float),
                  cmap='Greys', vmin=0, vmax=1, shading='auto')
ax[1].contour(RR / Rs, ZZ / Rs, rho, levels=[0.0],
              colors='red', linewidths=2)
ax[1].set_xlabel(r'$R / R_s$')
ax[1].set_ylabel(r'$z / R_s$')
ax[1].set_title(rf'Black = region with $\rho<0$, $q={q}$')

for a in ax:
    a.set_aspect('equal')

plt.tight_layout()
out = f'./Plots/flattened_nfw_negative_density_q{q}.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'\nSaved: {out}')
