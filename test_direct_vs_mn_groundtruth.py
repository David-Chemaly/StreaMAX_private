"""
TWO-STEP ground-truth validation:

Step 1: Validate our direct 3D Poisson integral code against a TRUE ground
        truth -- the Miyamoto-Nagai disk, which has BOTH an analytic
        closed-form Phi_MN(R, z) AND an analytic closed-form rho_MN(R, z).
        Plug rho_MN into our direct Poisson integral and check that the
        recovered Phi matches the analytic Phi_MN.

Step 2: Use the now-validated direct integral as a reference to check our
        flat-density NFW Chandrasekhar code at q = 0.5.

If Step 1 passes, we trust the direct integral. If Step 2 then matches,
we transitively trust the Chandrasekhar code.
"""

import numpy as np
from scipy.special import ellipk

from potentials import NFWPotentialFlatDensity
from constants import G

# 2D Gauss-Legendre nodes on [0, 1]
_NGL = 200
_x, _w = np.polynomial.legendre.leggauss(_NGL)
_U = 0.5 * (_x + 1.0)
_W = 0.5 * _w


# ---------------------------------------------------------------------------
# Miyamoto-Nagai disk: both Phi and rho closed form.
# ---------------------------------------------------------------------------
def phi_MN_analytic(R, z, M, a, b):
    """Miyamoto-Nagai 1975 potential."""
    zeta = np.sqrt(z*z + b*b)
    return -G * M / np.sqrt(R*R + (a + zeta)**2)

def rho_MN_analytic(R, z, M, a, b):
    """Miyamoto-Nagai 1975 density."""
    zeta = np.sqrt(z*z + b*b)
    num = a * R*R + (a + 3*zeta) * (a + zeta)**2
    den = (R*R + (a + zeta)**2)**(5/2) * zeta**3
    return (b*b * M / (4 * np.pi)) * num / den


# ---------------------------------------------------------------------------
# Direct 3D Poisson integral for an arbitrary axisymmetric rho(R, z).
# Same code as before, but now rho is a callable, not hardcoded NFW.
# ---------------------------------------------------------------------------
def phi_direct(R_obs, z_obs, rho_func):
    """
    Phi(R, z) = -4G integral R' dR' dz' rho(R', z') K(k) / sqrt((R+R')^2+(z-z')^2)
    with k^2 = 4 R R' / [(R+R')^2 + (z-z')^2]
    Computed on a 200x200 GL grid covering the full (R', z') plane via the
    change-of-variable R' = Rs0 * u/(1-u), z' = Rs0 * tan(pi(u-1/2)).
    The length scale Rs0 controls where the GL nodes concentrate -- pick
    something on the order of where rho is largest.
    """
    Rs0 = 5.0  # kpc -- typical scale for our test problems

    uR = _U[:, None]; uz = _U[None, :]
    wR = _W[:, None]; wz = _W[None, :]

    Rp     = Rs0 * uR / (1.0 - uR)
    dRp_du = Rs0 / (1.0 - uR)**2
    zp_arg = np.pi * (uz - 0.5)
    zp     = Rs0 * np.tan(zp_arg)
    dzp_du = Rs0 * np.pi / np.cos(zp_arg)**2

    denom2 = (R_obs + Rp)**2 + (z_obs - zp)**2
    denom  = np.sqrt(denom2)
    k2     = 4.0 * R_obs * Rp / denom2
    K      = ellipk(k2)

    rho = rho_func(Rp, zp)
    integrand = Rp * rho * K / denom * dRp_du * dzp_du

    return -4.0 * G * np.sum(wR * wz * integrand)


# ===========================================================================
# STEP 1 -- Validate direct integral against Miyamoto-Nagai ground truth.
# ===========================================================================
print("=" * 70)
print("STEP 1: Validate the direct 3D Poisson integral against MN ground truth")
print("=" * 70)
print("  rho_MN has analytic Phi_MN.  Plug rho_MN into our direct integral")
print("  and check it recovers Phi_MN. This is a TRUE ground-truth test.")
print()

# Pick a fairly flat MN disk: b << a makes it disk-like, b ~ a makes it
# Plummer-like spherical. Try both regimes.
for (M_MN, a_MN, b_MN, label) in [
    (1e11, 3.0, 0.3, "thin disk  (b/a = 0.1)"),
    (1e11, 3.0, 1.0, "fat disk   (b/a = 0.3)"),
    (1e11, 3.0, 3.0, "Plummer-ish (b/a = 1.0)"),
]:
    print(f"  MN model: M={M_MN:.1e}, a={a_MN}, b={b_MN} -- {label}")
    print(f"  {'(R, z)':>12s}  {'Phi_MN analytic':>17s}  {'Phi_direct':>15s}  {'rel err':>10s}")

    rho_callable = lambda R, z, M=M_MN, a=a_MN, b=b_MN: rho_MN_analytic(R, z, M, a, b)
    worst_rel = 0.0
    for (R_, z_) in [(1., 0.1), (3., 0.5), (3., 3.), (8., 5.), (15., 10.)]:
        p_ana    = phi_MN_analytic(R_, z_, M_MN, a_MN, b_MN)
        p_direct = phi_direct(R_, z_, rho_callable)
        rel = abs(p_ana - p_direct) / abs(p_ana)
        worst_rel = max(worst_rel, rel)
        print(f"  ({R_:4.1f},{z_:4.1f})   {p_ana:17.5e}  {p_direct:15.5e}  {rel:.2e}")
    print(f"  worst rel err: {worst_rel:.2e}")
    print()


# ===========================================================================
# STEP 2 -- Use the validated direct integral as a reference for our
#           flat-density NFW Chandrasekhar code at q = 0.5.
# ===========================================================================
print("=" * 70)
print("STEP 2: Validate our Chandrasekhar NFW (q=0.5) against the direct integral")
print("=" * 70)
print("  Now that we trust the direct integral (Step 1), use it as the")
print("  reference for our flat-density NFW at q=0.5.")
print()

logM = 12.0
Rs   = 15.0
M_phys = 10**logM
rho_s = M_phys / (4.0 * np.pi * Rs**3)
dirx, diry, dirz = 0.0, 0.0, 1.0

def rho_nfw_flat(R_, z_, q_):
    m = np.sqrt(R_**2 + (z_ / q_)**2)
    s = m / Rs
    return rho_s / (s * (1.0 + s)**2)

q_ = 0.5
rho_nfw_callable = lambda R, z: rho_nfw_flat(R, z, q_)

# Direct integral for NFW needs a larger length scale (Rs=15)
def phi_direct_with_scale(R_obs, z_obs, rho_func, Rs0):
    uR = _U[:, None]; uz = _U[None, :]
    wR = _W[:, None]; wz = _W[None, :]
    Rp     = Rs0 * uR / (1.0 - uR)
    dRp_du = Rs0 / (1.0 - uR)**2
    zp_arg = np.pi * (uz - 0.5)
    zp     = Rs0 * np.tan(zp_arg)
    dzp_du = Rs0 * np.pi / np.cos(zp_arg)**2
    denom2 = (R_obs + Rp)**2 + (z_obs - zp)**2
    denom  = np.sqrt(denom2)
    k2     = 4.0 * R_obs * Rp / denom2
    K      = ellipk(k2)
    rho    = rho_func(Rp, zp)
    integrand = Rp * rho * K / denom * dRp_du * dzp_du
    return -4.0 * G * np.sum(wR * wz * integrand)

print(f"  {'(R, z) [kpc]':>13s}  {'Phi_ours':>15s}  {'Phi_direct':>15s}  {'rel err':>10s}")
for (R_, z_) in [(5., 0.5), (15., 5.), (30., 10.), (5., 20.), (50., 50.)]:
    p_ours   = float(NFWPotentialFlatDensity(R_, 0.0, z_,
                                              logM, Rs, q_,
                                              dirx, diry, dirz))
    p_direct = phi_direct_with_scale(R_, z_, rho_nfw_callable, Rs)
    rel = abs(p_ours - p_direct) / abs(p_direct)
    print(f"  ({R_:5.1f},{z_:5.1f})  {p_ours:15.5e}  {p_direct:15.5e}  {rel:.2e}")
