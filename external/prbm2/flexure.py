from typing import NamedTuple
from beartype import beartype
import jax.numpy as jnp
from jaxtyping import jaxtyped

from prbm2.types_ import BodyState, FlexureDef, State, Mat33, Params, Vec3, Scalar
from prbm2.math import T_mat, angle_between, rot_mat

@jaxtyped(typechecker=beartype)
def get_body(q: State, i: int) -> BodyState:
    if(i == 0):
        return BodyState(jnp.zeros(3), jnp.zeros(3))
    else:
        return q[i-1] 

def create_flexure_def(q0: State, gamma: float, body_a: int, attach_a_l: Vec3, body_b: int, attach_b_l: Vec3) -> FlexureDef:
    body_a_ = get_body(q0, body_a)
    body_b_ = get_body(q0, body_b)

    rotmat_a = rot_mat(body_a_.rot)
    attach_a = body_a_.pos + rotmat_a @ attach_a_l
    rotmat_b = rot_mat(body_b_.rot)
    attach_b = body_b_.pos + rotmat_b @ attach_b_l

    flexure_v = attach_b - attach_a
    flexure_len = jnp.sqrt(jnp.dot(flexure_v, flexure_v))
    spring_len = flexure_len * gamma
    rig_len = flexure_len * (1-gamma)/2

    rig = flexure_v / flexure_len * rig_len
    spring_a = attach_a + rig
    spring_b = attach_b - rig

    return FlexureDef(
        body_a,
        rotmat_a.T @ (attach_a - body_a_.pos),
        rotmat_a.T @ (spring_a - body_a_.pos),
        body_b,
        rotmat_b.T @ (attach_b - body_b_.pos),
        rotmat_b.T @ (spring_b - body_b_.pos),
        
        rig_len,
        spring_len
    )


class FlexureGeom(NamedTuple):
    # Flexure: attach_a -(rig_a)-> spring_a -(spring_vec)-> spring_b -(rig_b)-> attach_b
    attach_a: Vec3 
    mass_a: Vec3  
    spring_a: Vec3  

    attach_b: Vec3
    mass_b: Vec3
    spring_b: Vec3

    rig_a: Vec3   
    rig_b: Vec3
    spring_v: Vec3

    rotmat_a: Mat33
    rotmat_b: Mat33

    spring_len: Scalar

def flexure_geom(q: State, p: Params, fd: FlexureDef) -> FlexureGeom:
    body_a = get_body(q, fd.body_a)
    body_b = get_body(q, fd.body_b)

    rotmat_a = rot_mat(body_a.rot)
    rotmat_b = rot_mat(body_b.rot)

    attach_a = body_a.pos + rotmat_a @ fd.attach_a_l
    attach_b = body_b.pos + rotmat_b @ fd.attach_b_l
    spring_a = body_a.pos + rotmat_a @ fd.spring_a_l
    spring_b = body_b.pos + rotmat_b @ fd.spring_b_l

    rig_a = spring_a - attach_a
    rig_b = attach_b - spring_b

    mass_a = attach_a + rig_a * p.mu
    mass_b = attach_b - rig_b * p.mu

    spring_v = spring_b - spring_a
    spring_len = jnp.sqrt(jnp.dot(spring_v, spring_v))

    return FlexureGeom(attach_a, mass_a, spring_a, attach_b, mass_b, spring_b, rig_a, rig_b, spring_v, rotmat_a, rotmat_b, spring_len)


def flexure_angles(f: FlexureGeom) -> tuple[Scalar, Scalar]:
    theta_a = angle_between(f.rig_a, f.spring_v)
    theta_b = angle_between(f.spring_v, f.rig_b)
    return (theta_a, theta_b)


class FlexureSpeeds(NamedTuple):
    v_mass_a: Vec3 
    w_mass_a: Vec3 
    v_mass_b: Vec3
    w_mass_b: Vec3
    v_spring: Vec3  
    w_spring: Vec3 