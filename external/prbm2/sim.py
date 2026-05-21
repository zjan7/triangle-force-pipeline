import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import optimistix as optx
from beartype import beartype
from jaxtyping import jaxtyped

from prbm2.flexure import flexure_angles, flexure_geom
from prbm2.math import rot_mat, skew, to_rod_local, T_mat
from prbm2.types_ import BodyState, FlexureDef, Mat33, Params, Scalar, State, Vec3, Wrench


# ---------------------------------------------------------------------------
# Energy terms
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def flexure_potential(q: State, p: Params, fd: FlexureDef) -> Scalar:
    f = flexure_geom(q, p, fd)

    rig_comp = (1-p.alpha)/2 * p.m_flex * (f.mass_a[2] + f.mass_b[2])
    spr_comp = p.alpha * p.m_flex * (f.spring_a[2] + f.spring_b[2])/2
    U_g = 9.81 * (rig_comp + spr_comp)

    theta_a, theta_b = flexure_angles(f)
    U_ex = 0.5 * p.k_ex * (f.spring_len - fd.spring_len0)**2
    U_th = 0.5 * p.k_th * (theta_a**2 + theta_b**2)

    return U_g + U_ex + U_th


@jaxtyped(typechecker=beartype)
def body_potential(q: State, p: Params, i: int) -> Scalar:
    body = q[i]
    return 9.81 * p.m_body * body.pos[2]


@jaxtyped(typechecker=beartype)
def potential(q: State, p: Params) -> Scalar:
    u: Scalar = jnp.zeros(())
    for i in range(2):
        u += body_potential(q, p, i)
    for fd in p.flexures:
        u += flexure_potential(q, p, fd)
    return u


@jaxtyped(typechecker=beartype)
def work(q: State, q0: State, wrench: Wrench) -> Scalar:
    """Virtual work done by a wrench applied to body 2.

    Uses small-rotation assumption: W = F·Δpos + τ·Δrot.
    """
    d_pos = q[1].pos - q0[1].pos
    d_rot = q[1].rot - q0[1].rot
    return jnp.dot(wrench.force, d_pos) + jnp.dot(wrench.torque, d_rot)


# ---------------------------------------------------------------------------
# Flatten / unflatten helpers
# ---------------------------------------------------------------------------

def state_to_vec(q: State) -> np.ndarray:
    """Flatten a State pytree to a 1-D numpy array."""
    leaves, _ = jax.tree_util.tree_flatten(q)
    return np.concatenate([np.asarray(l) for l in leaves])


def _make_vec_to_state(q_ref: State):
    """Return a vec_to_state closure bound to the tree structure of q_ref."""
    leaves0, treedef = jax.tree_util.tree_flatten(q_ref)
    split_at = np.cumsum([l.size for l in leaves0[:-1]]).tolist()

    def vec_to_state(x: np.ndarray) -> State:
        leaves = [jnp.array(c) for c in np.split(x, split_at)]
        return treedef.unflatten(leaves)

    return vec_to_state


# ---------------------------------------------------------------------------
# Minimisation
# ---------------------------------------------------------------------------

@jax.jit
def _obj_and_grad(q: State, q0: State, p: Params, wrench: Wrench):
    def _objective(q):
        return potential(q, p) - work(q, q0, wrench)
    return jax.value_and_grad(_objective)(q)


def minimize_energy(q0: State, p: Params, wrench: Wrench) -> tuple[State, object]:
    """
    Minimise  potential(q, p) − work(q, q0, wrench)  via L-BFGS-B.

    Returns the optimised State and the raw scipy OptimizeResult.
    """
    vec_to_state = _make_vec_to_state(q0)

    def flat_obj_grad(x: np.ndarray):
        q = vec_to_state(x)
        val, grad_q = _obj_and_grad(q, q0, p, wrench)
        grad_leaves, _ = jax.tree_util.tree_flatten(grad_q)
        return float(val), np.concatenate([np.asarray(g) for g in grad_leaves])

    x0 = state_to_vec(q0)
    result = minimize(
        flat_obj_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not result.success:
        print(f"Warning: optimizer did not converge: {result.message}")

    return vec_to_state(result.x), result


def _jax_objective(q: State, args) -> Scalar:
    q0, p, wrench = args
    return potential(q, p) - work(q, q0, wrench)

_jax_solver = optx.BFGS(rtol=1e-8, atol=1e-10)


def minimize_energy_jax(q0: State, p: Params, wrench: Wrench) -> tuple[State, object]:
    """
    Minimise  potential(q, p) − work(q, q0, wrench)  via optimistix BFGS.

    Operates directly on the State pytree — no flatten/unflatten round-trip.
    The entire solve is JIT-compiled end-to-end.

    Returns the optimised State and the optimistix Solution object.
    """
    sol = optx.minimise(
        _jax_objective,
        _jax_solver,
        q0,
        args=(q0, p, wrench),
        max_steps=1000,
        throw=False,
    )
    return sol.value, sol