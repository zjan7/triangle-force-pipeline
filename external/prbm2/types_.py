from jaxtyping import Array, Float
from typing import NamedTuple, TypeAlias
from jax import tree_util

# Shapes are annotated as jaxtyping dimension strings.
Vec3 = Float[Array, "3"]                        # world/body 3-vector
Mat33: TypeAlias = Float[Array, "3 3"]          # rotation or skew matrix
MassMat: TypeAlias = Float[Array, "12 12"]
Scalar: TypeAlias = Float[Array, ""]

class Wrench(NamedTuple):
    force:  Vec3   # applied force  [N]
    torque: Vec3   # applied torque [N·m]


class BodyState(NamedTuple):
    pos: Vec3   
    rot: Vec3    

# State + derivative for body 1 and 2, body 0 is always assumed to be fully zeroed.
State: TypeAlias = tuple[BodyState, BodyState] # Body states [1-2]

class FlexureDef(NamedTuple):
    body_a: int
    attach_a_l: Vec3
    spring_a_l: Vec3
    body_b: int
    attach_b_l: Vec3
    spring_b_l: Vec3

    rig_len: Scalar
    spring_len0: Scalar


class Params(NamedTuple):
    # Material
    e_mod: float  # Young's modulus
    v_rat: float  # Poisson's ratio
    rho: float  # density

    # PRB
    k_th: float
    k_ex: float
    gamma: float
    alpha: float
    mu: float

    # Structural
    m_body: float
    I_body: Vec3
    m_flex: float
    I_flex_rig: Vec3
    I_flex_spr: Vec3
    r_attach: float  # flexure attachment radius
    r_flex: float  # flexure radius

    # Initial condition stuff
    flexures: tuple[FlexureDef, ...]

def _params_flatten(p: Params):
    children = (
        p.e_mod, p.v_rat, p.rho, p.k_th, p.k_ex,
        p.gamma, p.alpha, p.mu, p.m_body, p.I_body,
        p.m_flex, p.I_flex_rig, p.I_flex_spr,
        p.r_attach, p.r_flex,
        p.flexures,  # FlexureDefs as leaves
    )
    aux = None  # nothing truly static
    return children, aux

def _params_unflatten(aux, children):
    return Params(*children)

tree_util.register_pytree_node(Params, _params_flatten, _params_unflatten)

def _flexuredef_flatten(fd: FlexureDef):
    children = (fd.attach_a_l, fd.spring_a_l, fd.attach_b_l, fd.spring_b_l, fd.rig_len, fd.spring_len0)
    aux = (fd.body_a, fd.body_b)
    return children, aux

def _flexuredef_unflatten(aux, children):
    body_a, body_b = aux
    attach_a_l, spring_a_l, attach_b_l, spring_b_l, rig_len, spring_len0 = children
    return FlexureDef(body_a, attach_a_l, spring_a_l, body_b, attach_b_l, spring_b_l, rig_len, spring_len0)

tree_util.register_pytree_node(FlexureDef, _flexuredef_flatten, _flexuredef_unflatten)