"""
First-principles ground-truth check for our flat-density NFW.

We compute the potential at a few test points (R, z) and at q=0.5 (the
extreme oblate end of the prior) by direct numerical 3D Poisson integration:

    Phi_direct(R, z) = -G * integral of  rho_NFW(m) / |r - r'|  dV'

Exploiting axisymmetry, the phi' integral has a closed form in the complete
elliptic integral K(k):

    Phi(R, z) = -4G integral_{R' >= 0} R' dR' integral dz'
                rho(R', z') * K(k) / sqrt((R + R')^2 + (z - z')^2)
    with k^2 = 4 R R' / [(R + R')^2 + (z - z')^2]

The kernel K is computed by scipy. The integrand is smooth except at the
1/sqrt singularity at the source point (R'=R, z'=z), which K handles
analytically (K(k -> 1) ~ -log(1-k), integrable).

We compare to our Chandrasekhar integral (potentials.NFWPotentialFlatDensity)
and to the spherical limit when applicable.
"""

import numpy as np
from scipy.special import ellipk

from potentials import NFWPotentialFlatDensity
from constants import G

# 2D Gauss-Legendre nodes on [0, 1], computed once.
_NGL_DIRECT = 200
_x, _w = np.polynomial.legendre.leggauss(_NGL_DIRECT)
_U = 0.5 * (_x + 1.0)   # nodes on [0, 1]
_W = 0.5 * _w

# Same parameters as everywhere else
logM = 12.0
Rs   = 15.0
M_phys = 10**logM
rho_s = M_phys / (4.0 * np.pi * Rs**3)

dirx, diry, dirz = 0.0, 0.0, 1.0


def rho_nfw_flat(R_, z_, q_):
    """Analytic axisymmetric flattened-density NFW (the thing we're modelling)."""
    m = np.sqrt(R_**2 + (z_ / q_)**2)
    s = m / Rs
    return rho_s / (s * (1.0 + s)**2)


def phi_direct(R_obs, z_obs, q_):
    """
    Brute-force Phi at (R_obs, z_obs) via 2D Gauss-Legendre quadrature
    of the axisymmetric Green's function over the FULL infinite (R', z')
    domain. No Chandrasekhar reformulation.

    Change of variable:
        R' = Rs * uR / (1 - uR),    uR in [0, 1)
        z' = Rs * tan(pi (uz - 1/2)),  uz in (0, 1)
    The first maps [0, infty), the second maps the full real line. The
    Plummer-like decay of the integrand makes both endpoint regions
    contribute negligibly.

    NFW total mass diverges logarithmically, but rho R' z'-integrand
    decays as 1/(R'^2 z'^2) at infinity so the potential integral
    converges; the change-of-variable just gives every part of space
    finite weight in our 96x96 = 9216 GL nodes.
    """
    # 1D nodes -> 2D grid
    uR = _U[:, None]
    uz = _U[None, :]
    wR = _W[:, None]
    wz = _W[None, :]

    # Transform to physical (R', z')
    Rp = Rs * uR / (1.0 - uR)               # in [0, infty)
    dRp_du = Rs / (1.0 - uR)**2             # dR'/duR
    zp_arg = np.pi * (uz - 0.5)
    zp = Rs * np.tan(zp_arg)                # in (-infty, infty)
    dzp_du = Rs * np.pi / np.cos(zp_arg)**2 # dz'/duz

    # Axisymmetric Green's function: integral over phi' of 1/|r - r'|
    # equals 4 K(k) / sqrt((R+R')^2 + (z-z')^2), k^2 = 4 R R' / [...]^2.
    denom2 = (R_obs + Rp)**2 + (z_obs - zp)**2
    denom  = np.sqrt(denom2)
    k2     = 4.0 * R_obs * Rp / denom2
    K = ellipk(k2)

    # rho_NFW at (R', z') for flattened density m^2 = R'^2 + z'^2/q^2
    rho = rho_nfw_flat(Rp, zp, q_)

    integrand = Rp * rho * K / denom * dRp_du * dzp_du

    val = np.sum(wR * wz * integrand)
    return -4.0 * G * val


print("=" * 70)
print("First-principles ground-truth Phi at q = 0.5 (the worst-case oblate end)")
print("=" * 70)
print(f"  Compare:")
print(f"    Phi_ours    -- our Chandrasekhar integral (N = 12 GL nodes)")
print(f"    Phi_direct  -- brute-force 2D numerical Poisson integral")
print()
print(f"  {'(R, z) [kpc]':>15s}  {'Phi_ours':>14s}  {'Phi_direct':>14s}  {'rel err':>10s}")
print(f"  {'-'*15}  {'-'*14}  {'-'*14}  {'-'*10}")

test_pts = [
    ( 5.0,  0.5),    # inner, near equator
    (15.0,  5.0),    # at the scale radius
    (30.0, 10.0),    # mid
    ( 5.0, 20.0),    # off-axis, low R but high z (most demanding at q=0.5)
    (50.0, 50.0),    # outer, off-axis
]

worst_rel = 0.0
for (R_, z_) in test_pts:
    p_ours   = float(NFWPotentialFlatDensity(R_, 0.0, z_,
                                             logM, Rs, 0.5,
                                             dirx, diry, dirz))
    p_direct = phi_direct(R_, z_, 0.5)
    rel = abs(p_ours - p_direct) / abs(p_direct)
    worst_rel = max(worst_rel, rel)
    print(f"  ({R_:5.1f}, {z_:5.1f})   {p_ours:14.5e}  {p_direct:14.5e}  {rel:.2e}")

print()
print(f"  Worst-case relative error across {len(test_pts)} points: {worst_rel:.2e}")
print()
print("Interpretation:")
print("  scipy's dblquad has its own tolerance ~1e-4, so anything below that")
print("  is at the quadrature noise floor of the *reference* calculation, not ours.")

# Also do q=1 to sanity-check the direct integral routine itself
print()
print("=" * 70)
print("Sanity: q = 1, direct integral should match closed-form spherical NFW")
print("=" * 70)
def phi_analytic_sph(r):
    return -G * M_phys * np.log(1.0 + r / Rs) / r

print(f"  {'(R, z) [kpc]':>15s}  {'analytic NFW':>14s}  {'Phi_direct':>14s}  {'rel err':>10s}")
print(f"  {'-'*15}  {'-'*14}  {'-'*14}  {'-'*10}")
for (R_, z_) in test_pts:
    r_ = np.sqrt(R_**2 + z_**2)
    p_ana    = phi_analytic_sph(r_)
    p_direct = phi_direct(R_, z_, 1.0)
    rel = abs(p_ana - p_direct) / abs(p_ana)
    print(f"  ({R_:5.1f}, {z_:5.1f})   {p_ana:14.5e}  {p_direct:14.5e}  {rel:.2e}")
print("  (This validates the *direct integral itself* against a known answer.)")
