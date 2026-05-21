"""
Meshcat visualization for prbdm simulation results.

Usage:
    from prbdm.viz import visualize, visualize_pose, visualize_comparison
    visualize(sim_result, p)
    visualize_pose(q0, p)
    visualize_comparison(q0, q_opt, p)   # initial (grey) + optimized (blue)

Then open the URL printed to the terminal in your browser.
Call viz.start_recording() / viz.stop_recording() if you want to export.
"""

import time
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from prbm2.flexure import flexure_geom
from prbm2.types_ import BodyState, Params, State


# ── Geometry constants ────────────────────────────────────────────────────────

BODY_COLOR    = 0x4488ff
FLEXURE_COLOR = 0xffaa33
MASS_COLOR    = 0xff4444
GROUND_COLOR  = 0x888888

RIG_COLOR    = 0xffaa33  # orange
SPRING_COLOR = 0x44ff88  # green

# Ghost colors (initial/reference pose shown alongside optimized)
BODY_COLOR_GHOST   = 0xaaaaaa  # grey
RIG_COLOR_GHOST    = 0xccaa77
SPRING_COLOR_GHOST = 0x88ccaa
MASS_COLOR_GHOST   = 0xcc8888

BODY_HEIGHT   = 0.004   # disc thickness
MASS_RADIUS   = 0.003   # sphere radius for PRB lumped masses
ROD_RADIUS    = 0.002   # cylinder radius for flexure rods


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rod_transform(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """
    4×4 transform that orients and positions a cylinder between p_a and p_b.
    The cylinder geometry must already have the correct length baked in.
    Meshcat cylinders are centred at origin and aligned along Y by default.
    """
    diff   = p_b - p_a
    length = np.linalg.norm(diff)
    if length < 1e-9:
        return tf.translation_matrix(p_a)

    mid   = (p_a + p_b) / 2
    y_hat = diff / length

    ref   = np.array([1.0, 0.0, 0.0]) if abs(y_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_hat = np.cross(ref, y_hat); x_hat /= np.linalg.norm(x_hat)
    z_hat = np.cross(x_hat, y_hat)

    R = np.eye(4)
    R[:3, 0] = x_hat
    R[:3, 1] = y_hat
    R[:3, 2] = z_hat
    R[:3, 3] = mid
    return R


def _body_transform(pos: np.ndarray, rot_vec: np.ndarray) -> np.ndarray:
    """4×4 transform for a body given position and Rodrigues rotation vector."""
    theta = np.linalg.norm(rot_vec)
    R = np.eye(3)
    if theta > 1e-9:
        k = rot_vec / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    # rotate cylinder from Y-up to Z-up so disc lies in XY plane
    T = T @ tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    return T


# ── Scene setup ───────────────────────────────────────────────────────────────

def _setup_scene(
    vis: meshcat.Visualizer,
    p: Params,
    ns: str = "",
    body_color: int = BODY_COLOR,
    rig_color: int = RIG_COLOR,
    spring_color: int = SPRING_COLOR,
    mass_color: int = MASS_COLOR,
    opacity: float = 0.5,
) -> None:
    """Create all persistent geometry objects under namespace *ns*."""
    prefix = f"{ns}/" if ns else ""
    body_mat = g.MeshLambertMaterial(color=body_color, opacity=opacity, transparent=True)

    if not ns:
        # Only add ground for the primary (non-ghost) scene
        vis["ground"].set_object(
            g.Box([0.3, 0.3, 0.001]),
            g.MeshLambertMaterial(color=GROUND_COLOR, opacity=0.4, transparent=True)
        )
        vis["ground"].set_transform(tf.translation_matrix([0, 0, -0.0005]))

    # Body 0 — fixed ground body
    vis[f"{prefix}body/0"].set_object(
        g.Cylinder(BODY_HEIGHT / 2, p.r_attach * 1.2),
        body_mat
    )
    vis[f"{prefix}body/0"].set_transform(tf.rotation_matrix(np.pi / 2, [1, 0, 0]))

    # Bodies 1 and 2
    for i in range(1, 3):
        vis[f"{prefix}body/{i}"].set_object(
            g.Cylinder(BODY_HEIGHT / 2, p.r_attach * 1.2),
            body_mat
        )

    # Flexure segments
    for fi, fd in enumerate(p.flexures):
        base = f"{prefix}flexure/{fi}"

        rig_len    = float(fd.rig_len)
        spring_len = float(fd.spring_len0)

        rig_mat    = g.MeshLambertMaterial(color=rig_color,    opacity=0.7, transparent=True)
        spring_mat = g.MeshLambertMaterial(color=spring_color, opacity=0.7, transparent=True)
        mass_mat_  = g.MeshLambertMaterial(color=mass_color,   opacity=0.9, transparent=True)

        vis[f"{base}/rig_a" ].set_object(g.Cylinder(rig_len,             ROD_RADIUS),       rig_mat)
        vis[f"{base}/spring"].set_object(g.Cylinder(spring_len,          ROD_RADIUS * 0.7), spring_mat)
        vis[f"{base}/rig_b" ].set_object(g.Cylinder(rig_len,             ROD_RADIUS),       rig_mat)
        vis[f"{base}/mass_a"].set_object(g.Sphere(MASS_RADIUS), mass_mat_)
        vis[f"{base}/mass_b"].set_object(g.Sphere(MASS_RADIUS), mass_mat_)


# ── Per-frame update ──────────────────────────────────────────────────────────

def _update_frame(vis: meshcat.Visualizer, q: State, p: Params, ns: str = "") -> None:
    """Push transforms for a single time step under namespace *ns*."""
    prefix = f"{ns}/" if ns else ""

    # Bodies
    for i, body in enumerate(q):
        pos = np.array(body.pos, dtype=float)
        rot = np.array(body.rot, dtype=float)
        vis[f"{prefix}body/{i+1}"].set_transform(_body_transform(pos, rot))

    # Flexures
    for fi, fd in enumerate(p.flexures):
        f    = flexure_geom(q, p, fd)
        base = f"{prefix}flexure/{fi}"

        a_a = np.array(f.attach_a, dtype=float)
        s_a = np.array(f.spring_a, dtype=float)
        s_b = np.array(f.spring_b, dtype=float)
        a_b = np.array(f.attach_b, dtype=float)
        m_a = np.array(f.mass_a,   dtype=float)
        m_b = np.array(f.mass_b,   dtype=float)

        vis[f"{base}/rig_a" ].set_transform(_rod_transform(a_a, s_a))
        vis[f"{base}/spring"].set_transform(_rod_transform(s_a, s_b))
        vis[f"{base}/rig_b" ].set_transform(_rod_transform(s_b, a_b))
        vis[f"{base}/mass_a"].set_transform(tf.translation_matrix(m_a))
        vis[f"{base}/mass_b"].set_transform(tf.translation_matrix(m_b))


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_pose(q: State, p: Params) -> meshcat.Visualizer:
    """Render a single pose and block until Enter is pressed."""
    vis = meshcat.Visualizer()
    vis.open()
    _setup_scene(vis, p)
    _update_frame(vis, q, p)
    input("Press Enter to exit...")
    return vis


def visualize_comparison(
    q_initial: State,
    q_result: State,
    p: Params,
) -> meshcat.Visualizer:
    """
    Render two poses simultaneously for comparison.

    The initial pose is shown as a faded grey ghost; the result pose is shown
    in the normal blue/orange/green colours.

    Parameters
    ----------
    q_initial : undeformed / reference state
    q_result  : optimized or otherwise modified state
    p         : Params
    """
    vis = meshcat.Visualizer()
    vis.open()

    # Ghost (initial) — grey, low opacity, under "ghost/" namespace
    _setup_scene(
        vis, p, ns="ghost",
        body_color=BODY_COLOR_GHOST,
        rig_color=RIG_COLOR_GHOST,
        spring_color=SPRING_COLOR_GHOST,
        mass_color=MASS_COLOR_GHOST,
        opacity=0.25,
    )
    _update_frame(vis, q_initial, p, ns="ghost")

    # Result — normal colours, under root namespace
    _setup_scene(vis, p)
    _update_frame(vis, q_result, p)

    print("Grey  = initial pose")
    print("Blue  = result pose")
    input("Press Enter to exit...")
    return vis
