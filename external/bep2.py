"""
solve_module.py

Drop-in replacement for the original solve_module() using the prbdm library.

solve_module_jax(force_vec, upside_down, q0, p)
    -> (c1, c2, c3) : three contact points on body 1 (middle disc / body B)
                      as jnp arrays of shape (3,)

The function is pure (no global state) so it is safe to jit/vmap.
Pass in the q0 and p built by run.py (or build_module()).

For convenience, build_module() constructs q0 and p from the same constants
used in run.py so you don't have to duplicate that setup.
"""

import jax
import jax.numpy as jnp
from jax.numpy import pi, cos, sin
import numpy as np

jax.config.update("jax_enable_x64", True)

from prbm2.flexure import create_flexure_def
from prbm2.math import rot_mat
from prbm2.sim import minimize_energy_jax
from prbm2.types_ import BodyState, FlexureDef, Params, State, Vec3, Wrench


mm = 1e-3

depth    = 27.5 * mm #hoogte bovenplaat tot onderplaat
r_attach = 9.32 * mm #afstand tussen middelpunt driehoek naar zijkant waar flexure start
z        = 33.75 * mm   #lengte staaf midden naar uiteinden 
l        = 7 *mm #afstand midden van detectie driehoek naar punt

t     =  1.25* mm #radius flexure
E     =26e6 #E modulusI     
I = pi * (2 * t)**4 / 64 #second moment of inertia ronde balk
A_sec = pi * t**2 

gamma    = 0.85 #moeten nog kloppen uit literatuur
kappa_th = 2.65 #moeten nog kloppen uit literatuur

_chord = 2 * r_attach * float(jnp.sin(pi / 3))  # in-plane chord between attachment points
L0     = float(jnp.sqrt(_chord**2 + (depth / 2)**2))  # actual 3D flexure length
k_th = gamma * kappa_th * E * I / L0
k_ex = E * A_sec / (gamma * L0)

mu    = 0.5 * (1 - gamma)
alpha = 0.5
rho   = 1200.0 #materiaal dichtheid
m_flex = rho * A_sec * L0
m_body = 0.0


# ---------------------------------------------------------------------------
# build_module() — construct the reference q0 and Params once
# ---------------------------------------------------------------------------
def build_module() -> tuple[State, Params]:
    """Return (q0, p) for the reference three-flexure module."""

    q0: State = (
        BodyState(jnp.array([0.0, 0.0, depth / 2]), jnp.zeros(3)),
        BodyState(jnp.array([0.0, 0.0, depth      ]), jnp.zeros(3)),
    )

    def attach(r: float, frac: float) -> Vec3:
        θ = frac * 2 * pi
        return jnp.array([r * cos(θ), r * sin(θ), 0.0])

    flexures: list[FlexureDef] = []
    for i in range(3):
        # ground (0) → middle disc (1)
        flexures.append(create_flexure_def(
            q0, gamma,
            body_a=0, attach_a_l=attach(r_attach,  i      / 3),
            body_b=1, attach_b_l=attach(r_attach, (i + 1) / 3),
        ))
        # top disc (2) → middle disc (1)
        flexures.append(create_flexure_def(
            q0, gamma,
            body_a=2, attach_a_l=attach(r_attach, (i - 0.5) / 3),
            body_b=1, attach_b_l=attach(r_attach, (i + 0.5) / 3),
        ))

    p = Params(
        e_mod = E,
        v_rat = 0.3,
        rho   = rho,
        k_th  = k_th,
        k_ex  = k_ex,
        gamma = gamma,
        mu    = mu,
        alpha = alpha,
        m_body = m_body,
        m_flex = m_flex,
        I_body     = jnp.array([1e-8, 1e-8, 1e-8]),
        I_flex_rig = jnp.array([1e-8, 1e-8, 1e-8]),
        I_flex_spr = jnp.array([1e-8, 1e-8, 1e-8]),
        r_flex   = t,
        r_attach = r_attach,
        flexures = tuple(flexures),
    )

    return q0, p


# ---------------------------------------------------------------------------
# Contact-point local coordinates
# ---------------------------------------------------------------------------
def _contact_points_local(upside_down: bool) -> tuple[Vec3, Vec3, Vec3]:
    """
    Three contact points in body-1's local frame.

    Original logic
    --------------
    The triangle lies at z = -depth/2 in the body frame (halfway toward the
    ground body).  The two orientations are mirror images about the XY plane
    of the disc:

      upside_down=False  (normal):
        c1 at angle  270°  ( 0, -l, -depth/2 )
        c2 at angle  150°  ( -l·cos30°,  l·sin30°, -depth/2 )
        c3 at angle   30°  (  l·cos30°,  l·sin30°, -depth/2 )

      upside_down=True:
        c1 at angle   90°  ( 0,  l, -depth/2 )
        c2 at angle  210°  ( -l·cos30°, -l·sin30°, -depth/2 )
        c3 at angle  330°  (  l·cos30°, -l·sin30°, -depth/2 )
    """
    c30 = float(jnp.cos(pi / 6))
    s30 = float(jnp.sin(pi / 6))  # = 0.5

    if upside_down:
        p1 = jnp.array([ 0.0,          l,  z])
        p2 = jnp.array([-l * c30, -l * s30, z])
        p3 = jnp.array([ l * c30, -l * s30, z])
    else:
        p1 = jnp.array([ 0.0,         -l,  z])
        p2 = jnp.array([-l * c30,  l * s30, z])
        p3 = jnp.array([ l * c30,  l * s30, z])

    return p1, p2, p3


# ---------------------------------------------------------------------------
# Module-level constants — built once at import time
# ---------------------------------------------------------------------------
_q0, _p = build_module()


# ---------------------------------------------------------------------------
# solve_module_jax
# ---------------------------------------------------------------------------
def solve_module(
    force_vec,
    upside_down: bool,
) -> tuple[Vec3, Vec3, Vec3]:
    """
    Solve the module equilibrium under *force_vec* applied to body 2 (top disc)
    and return three contact points on body 1 (middle disc) in world coordinates.

    Parameters
    ----------
    force_vec   : (3,) force applied to body 2 [N]
    upside_down : flips the contact-triangle orientation (see original script)

    Returns
    -------
    (c1, c2, c3) : three Vec3 world-frame contact points on body 1
    """
    wrench = Wrench(force=jnp.array(force_vec), torque=jnp.zeros(3))
    q_opt, _ = minimize_energy_jax(_q0, _p, wrench)

    # Body 1 = middle disc (index 0 in the State tuple, which holds bodies 1 and 2)
    body1 = q_opt[0]
    R = rot_mat(body1.rot)

    p1_l, p2_l, p3_l = _contact_points_local(upside_down)

    c1 = body1.pos + R @ p1_l
    c2 = body1.pos + R @ p2_l
    c3 = body1.pos + R @ p3_l

    return c1, c2, c3


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Zero force — should return undeformed contact points
    f0 = jnp.zeros(3)
    c1, c2, c3 = solve_module_jax(f0, False)
    print("Zero force (upside_down=False):")
    print(f"  c1 = {np.array(c1).round(6)}")
    print(f"  c2 = {np.array(c2).round(6)}")
    print(f"  c3 = {np.array(c3).round(6)}")

    # Small lateral force
    f1 = jnp.array([0.05, 0.0, 0.0])
    c1, c2, c3 = solve_module_jax(f1, False)
    print("\n0.05 N lateral (upside_down=False):")
    print(f"  c1 = {np.array(c1).round(6)}")
    print(f"  c2 = {np.array(c2).round(6)}")
    print(f"  c3 = {np.array(c3).round(6)}")

    # Upside-down variant
    c1, c2, c3 = solve_module_jax(f1, True)
    print("\n0.05 N lateral (upside_down=True):")
    print(f"  c1 = {np.array(c1).round(6)}")
    print(f"  c2 = {np.array(c2).round(6)}")
    print(f"  c3 = {np.array(c3).round(6)}")