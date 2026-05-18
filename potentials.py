import jax
import jax.numpy as jnp
import numpy as _np
# jax.config.update("jax_enable_x64", True)

from flax import struct
import functools # Import functools for partial methods

from utils import get_mat

from constants import G, EPSILON

# Gauss-Legendre nodes/weights for the flat-density NFW Chandrasekhar
# integral. After the substitution v = cos(theta), the integrand is a
# smooth rational function on [0, 1] with no singularities, so 32 nodes
# are enough for ~1e-7 relative accuracy. Computed once at import.
_N_QUAD_FLATDENS = 12
_x_gl, _w_gl = _np.polynomial.legendre.leggauss(_N_QUAD_FLATDENS)
_V_GL = jnp.asarray(0.5 * (_x_gl + 1.0))   # nodes on [0, 1]
_W_GL = jnp.asarray(0.5 * _w_gl)            # weights on [0, 1]

### NFW Functions ###
@jax.jit
def NFWPotential(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the NFW potential at a given position (x, y, z) with specified parameters.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        float: The computed potential at the given position.
    """
    r_input = jnp.array([x, y, z])
    
    # Get rotation matrix
    rot_mat = get_mat(dirx, diry, dirz)
    
    r_vect = jnp.dot(rot_mat, r_input)
    rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]
    
    r = jnp.sqrt(rx**2 + ry**2 + (rz / q)**2 + EPSILON)
    
    phi = -G * 10**logM / r * jnp.log(1 + r / Rs)
    return phi  # kpc²/Gyr²

@jax.jit
def NFWAcceleration(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the acceleration as the negative gradient of the NFW potential.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        jnp.ndarray: The acceleration vector at the given position.
    """
    potential_func = lambda pos: NFWPotential(pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    
    acc = jax.grad(potential_func)(jnp.array([x, y, z]))
    return -acc # kpc / Gyr2

@jax.jit
def NFWHessian(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the Hessian matrix of the NFW potential at a given position.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: NFWPotential(pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    
    hess = jax.hessian(potential_func)(jnp.array([x, y, z]))
    return hess # 1/Gyr2

@jax.jit
def NFWdHessian(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Computes the derivative of the Hessian matrix of the NFW potential at a given position.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        q (float): Axis ratio.
        dirx (float): x-component of the direction vector.
        diry (float): y-component of the direction vector.
        dirz (float): z-component of the direction vector.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: NFWPotential(pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)

    dhess = jax.jacfwd(jax.hessian(potential_func))(jnp.array([x, y, z]))
    return dhess # 1/Gyr2/kpc

### NFW with flattening applied to the DENSITY (rho = rho_NFW(m), m^2 = R^2 + z^2/q^2) ###
# The potential is then recovered via Chandrasekhar's homoeoidal integral,
# guaranteeing rho > 0 everywhere for any q > 0 (unlike potential-flattening,
# which breaks positivity outside q ~ [0.7, 1.3]).

@jax.jit
def NFWPotentialFlatDensity(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Potential of an axisymmetric NFW with the flattening applied to the
    *density* rather than the potential. Same call signature as NFWPotential.

    rho(R, z) = rho_NFW(m),  m^2 = R^2 + z^2 / q^2.
    Phi(R, z) = -(G M q^2 / R_s) * I(R, z, q)
    where I is the dimensionless 1D Chandrasekhar integral, written with
    the substitution v = cos(theta) so no transcendentals appear:
       I = integral_{v=0..1} dv / [(1 + m_tilde(v)) * D(v)],
       D(v)       = q^2 + (1 - q^2) v^2,
       m_tilde(v) = v * sqrt(xi^2 / D(v) + zeta^2 / q^2),
       xi  = R / R_s,   zeta = z / R_s.
    """
    r_input = jnp.array([x, y, z])
    rot_mat = get_mat(dirx, diry, dirz)
    r_vect = jnp.dot(rot_mat, r_input)
    rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]

    R = jnp.sqrt(rx**2 + ry**2 + EPSILON)
    Z = rz

    xi   = R / Rs
    zeta = Z / Rs

    v   = _V_GL
    q2  = q * q
    D   = q2 + (1.0 - q2) * v * v          # cos^2 + q^2 sin^2
    m   = v * jnp.sqrt(xi * xi / D + zeta * zeta / q2 + EPSILON)

    integrand = 1.0 / ((1.0 + m) * D)
    integral  = jnp.sum(_W_GL * integrand)

    phi = -(G * 10**logM * q2 / Rs) * integral
    return phi  # kpc^2 / Gyr^2


@jax.jit
def NFWAccelerationFlatDensity(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """
    Acceleration = -grad(NFWPotentialFlatDensity), computed analytically
    (no autodiff). The two cylindrical derivatives reduce to two quadratures
    on the same Gauss-Legendre nodes used for Phi:
        dI/dxi   = -xi      * sum_v w v^2 / [(1+m)^2 D^2 m]
        dI/dzeta = -zeta/q2 * sum_v w v^2 / [(1+m)^2 D   m]
    Then rotated back to world coordinates by the same rotation as in Phi.
    """
    r_input = jnp.array([x, y, z])
    rot_mat = get_mat(dirx, diry, dirz)
    r_vect = jnp.dot(rot_mat, r_input)
    rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]

    R2 = rx * rx + ry * ry + EPSILON
    R  = jnp.sqrt(R2)
    Z  = rz

    xi   = R / Rs
    zeta = Z / Rs

    v   = _V_GL
    v2  = v * v
    q2  = q * q
    D   = q2 + (1.0 - q2) * v2
    A   = xi * xi / D + zeta * zeta / q2 + EPSILON
    m   = v * jnp.sqrt(A)

    base = v2 / ((1.0 + m) * (1.0 + m) * m + EPSILON)
    dI_dxi   = -xi          * jnp.sum(_W_GL * base / (D * D))
    dI_dzeta = -(zeta / q2) * jnp.sum(_W_GL * base / D)

    pref = -(G * 10**logM * q2 / Rs)
    # dPhi/dR = (1/Rs) dPhi/dxi  = (1/Rs) pref dI/dxi
    dPhi_dR = pref * dI_dxi   / Rs
    dPhi_dZ = pref * dI_dzeta / Rs

    # Cartesian force in the halo (rotated) frame
    F_halo = jnp.array([
        -dPhi_dR * rx / R,
        -dPhi_dR * ry / R,
        -dPhi_dZ,
    ])

    # Rotate back to world frame.  rot_mat sends world -> halo, so the
    # inverse (transpose, since it's orthonormal) sends halo -> world.
    return jnp.dot(rot_mat.T, F_halo)  # kpc / Gyr^2


@jax.jit
def NFWHessianFlatDensity(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """Hessian of NFWPotentialFlatDensity, via autodiff."""
    potential_func = lambda pos: NFWPotentialFlatDensity(
        pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    hess = jax.hessian(potential_func)(jnp.array([x, y, z]))
    return hess  # 1 / Gyr^2


@jax.jit
def NFWdHessianFlatDensity(x, y, z, logM, Rs, q, dirx, diry, dirz):
    """Jacobian of Hessian of NFWPotentialFlatDensity, via autodiff."""
    potential_func = lambda pos: NFWPotentialFlatDensity(
        pos[0], pos[1], pos[2], logM, Rs, q, dirx, diry, dirz)
    dhess = jax.jacfwd(jax.hessian(potential_func))(jnp.array([x, y, z]))
    return dhess  # 1 / Gyr^2 / kpc


### Plummer Functions ###
@jax.jit
def PlummerPotential(x, y, z, logM, Rs, x_origin=0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the Plummer potential at a given position (x, y, z) with specified parameters.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        float: The computed potential at the given position.
    """
    r = jnp.sqrt((x - x_origin)**2 + (y - y_origin)**2 + (z - z_origin)**2 + EPSILON)
    phi = -G * 10**logM / jnp.sqrt(r**2 + Rs**2)
    return phi # kpc²/Gyr²

@jax.jit
def PlummerAcceleration(x, y, z, logM, Rs, x_origin=0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the acceleration as the negative gradient of the Plummer potential.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        jnp.ndarray: The acceleration vector at the given position.
    """
    potential_func = lambda pos: PlummerPotential(pos[0], pos[1], pos[2], logM, Rs, x_origin, y_origin, z_origin)
    
    acc = jax.grad(potential_func)(jnp.array([x, y, z]))
    return -acc  # kpc / Gyr²

@jax.jit
def PlummerHessian(x, y, z, logM, Rs, x_origin= 0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the Hessian matrix of the Plummer potential at a given position.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: PlummerPotential(pos[0], pos[1], pos[2], logM, Rs, x_origin, y_origin, z_origin)
    
    hess = jax.hessian(potential_func)(jnp.array([x, y, z]))
    return hess # 1/Gyr²

@jax.jit
def PlummerdHessian(x, y, z, logM, Rs, x_origin= 0.0, y_origin=0.0, z_origin=0.0):
    """
    Computes the derivative of the Hessian matrix of the Plummer potential at a given position.
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.
        logM (float): Logarithm of the mass of the halo.
        Rs (float): Scale radius of the halo.
        x_origin (float): x-coordinate of the origin.
        y_origin (float): y-coordinate of the origin.
        z_origin (float): z-coordinate of the origin.

    Returns:
        jnp.ndarray: The Hessian matrix at the given position.
    """
    potential_func = lambda pos: PlummerPotential(pos[0], pos[1], pos[2], logM, Rs, x_origin, y_origin, z_origin)

    dhess = jax.jacfwd(jax.hessian(potential_func))(jnp.array([x, y, z]))
    return dhess # 1/Gyr²/kpc