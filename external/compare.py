"""
compare_solve_module.py

Compares the original solve_module() (pyRBM / scipy) against the new
solve_module_jax() (prbdm / optimistix) across a grid of test forces.

For each test case, prints the world-frame contact points from both
implementations and the absolute difference in each coordinate.
"""

import numpy as np
from numpy import pi, cos, sin

# ── new implementation ────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from bep2 import solve_module

# ── original implementation (inline) ─────────────────────────────────────────
from numpy import zeros, array, sin as nsin, cos as ncos, acos, dot
from numpy.linalg import norm
from scipy.optimize import minimize as sp_minimize

gamma_orig    = .85
kappa_theta   = 2.65
mm            = 1e-3
depth         = 20 * mm   # must match solve_module.py

def _angle(a, b, atol=1e-6):
    if norm(a - b) < atol:
        return 0
    return acos(np.clip(dot(a, b) / norm(a) / norm(b), -1, 1))

def _R(t):
    return array([[ncos(t), -nsin(t)], [nsin(t), ncos(t)]])

def _RX(t):
    return array([[1,0,0],[0,ncos(t),-nsin(t)],[0,nsin(t),ncos(t)]])
def _RY(t):
    return array([[ncos(t),0,nsin(t)],[0,1,0],[-nsin(t),0,ncos(t)]])
def _RZ(t):
    return array([[ncos(t),-nsin(t),0],[nsin(t),ncos(t),0],[0,0,1]])

def _rotmat3(angles):
    return _RX(angles[0]) @ _RY(angles[1]) @ _RZ(angles[2])

class _Flexure:
    def __init__(self, posA, apA, posB, apB):
        self.apA_l = array(apA)
        self.apB_l = array(apB)
        agA0 = posA + self.apA_l
        agB0 = posB + self.apB_l
        v0 = agB0 - agA0
        self.len0 = norm(v0)
        uv0 = v0 / self.len0
        spA0 = agA0 + (1 - gamma_orig) / 2 * v0
        spB0 = agB0 - (1 - gamma_orig) / 2 * v0
        self.springlen0 = norm(spB0 - spA0)
        self.spA_l = spA0 - posA
        self.spB_l = spB0 - posB
        self.uvA_l = uv0
        self.uvB_l = uv0
        # mutable
        self.agA = agA0; self.agB = agB0
        self.spA = spA0; self.spB = spB0
        self.uvA = uv0;  self.uvB = uv0

    def energy(self, A, E, I):
        kappa = gamma_orig * kappa_theta * E * I / self.len0
        k     = E * A / self.springlen0
        v  = self.spB - self.spA
        sl = norm(v)
        uv = v / sl
        tA = _angle(self.uvA, uv)
        tB = _angle(self.uvB, uv)
        return kappa*tA**2/2 + kappa*tB**2/2 + k*(sl - self.springlen0)**2/2

class _Body:
    def __init__(self, pos0):
        self.pos0   = array(pos0, dtype=float)
        self.pos    = array(pos0, dtype=float)
        self.angles = zeros(3)
        self.rotmat = np.eye(3)
        self.flexures = []; self.which = []
        self.forces   = []

    def move(self, pos, angles):
        self.pos    = array(pos)
        self.angles = array(angles)
        self.rotmat = _rotmat3(angles)
        for fl, w in zip(self.flexures, self.which):
            if w:
                fl.agA = self.rotmat @ fl.apA_l + self.pos
                fl.spA = self.rotmat @ fl.spA_l + self.pos
                fl.uvA = self.rotmat @ fl.uvA_l
            else:
                fl.agB = self.rotmat @ fl.apB_l + self.pos
                fl.spB = self.rotmat @ fl.spB_l + self.pos
                fl.uvB = self.rotmat @ fl.uvB_l

    def energy(self):
        e = 0
        for ap_l, fvec in self.forces:
            e -= dot(self.rotmat @ ap_l + self.pos, fvec)
        return e

def _init_orig():
    r = 8.773827 * mm
    n = 3
    bA = _Body([0, 0, 0])
    bB = _Body([0, 0, 0.5 * depth])
    bC = _Body([0, 0, depth])
    flexures = []
    for i in range(n):
        fl = _Flexure(bA.pos, (r*ncos(i/n*2*pi), r*nsin(i/n*2*pi), 0),
                      bB.pos, (r*ncos((i+1)/n*2*pi), r*nsin((i+1)/n*2*pi), 0))
        bA.flexures.append(fl); bA.which.append(True)
        bB.flexures.append(fl); bB.which.append(False)
        flexures.append(fl)
        fl2 = _Flexure(bC.pos, (r*ncos((i-.5)/n*2*pi), r*nsin((i-.5)/n*2*pi), 0),
                       bB.pos, (r*ncos((i+.5)/n*2*pi), r*nsin((i+.5)/n*2*pi), 0))
        bC.flexures.append(fl2); bC.which.append(True)
        bB.flexures.append(fl2); bB.which.append(False)
        flexures.append(fl2)
    return bA, bB, bC, flexures

def solve_module_orig(force_vec, upside_down):
    t_wire = mm
    A = pi * t_wire**2
    E = 850e6
    I = 0.1 * pi * (1e-3)**4 / 2

    bA, bB, bC, flexures = _init_orig()
    bC.forces.append((zeros(3), array(force_vec, dtype=float)))

    free = [bB, bC]

    def obj(x):
        for k, b in enumerate(free):
            b.move(x[6*k:6*k+3], x[6*k+3:6*k+6])
        e = sum(fl.energy(A, E, I) for fl in flexures)
        e += sum(b.energy() for b in free)
        return e

    x0 = np.concatenate([np.concatenate([b.pos, b.angles]) for b in free])
    res = sp_minimize(obj, x0, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    obj(res.x)   # restore body state to optimum

    l = 4 * mm
    R = bB.rotmat
    p = bB.pos
    z = -0.5 * depth
    c30 = ncos(pi / 6)
    s30 = nsin(pi / 6)
    if upside_down:
        c1 = R @ array([ 0,        l,   z]) + p
        c2 = R @ array([-l*c30,  -l*s30, z]) + p
        c3 = R @ array([ l*c30,  -l*s30, z]) + p
    else:
        c1 = R @ array([ 0,       -l,   z]) + p
        c2 = R @ array([-l*c30,  l*s30, z]) + p
        c3 = R @ array([ l*c30,  l*s30, z]) + p
    return c1, c2, c3


# ── Test cases ────────────────────────────────────────────────────────────────
test_cases = [
    # (label,               force_vec,          upside_down)
    ("zero force, normal",  [0.0,  0.0,  0.0],  False),
    ("zero force, flipped", [0.0,  0.0,  0.0],  True ),
    ("+x 0.05 N, normal",   [0.05, 0.0,  0.0],  False),
    ("+x 0.05 N, flipped",  [0.05, 0.0,  0.0],  True ),
    ("+y 0.05 N",           [0.0,  0.05, 0.0],  False),
    ("-z 0.05 N",           [0.0,  0.0, -0.05], False),
    ("diagonal 0.03 N",     [0.03, 0.03, 0.0],  False),
]

COL = 14   # column width for numbers

def fmt(v):
    return "  ".join(f"{x:+{COL}.8f}" for x in v)

def header(label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'':30s}  {'x':>{COL}}  {'y':>{COL}}  {'z':>{COL}}")

max_errors = []

for label, fvec, ud in test_cases:
    header(label)

    c1o, c2o, c3o = solve_module_orig(fvec, ud)
    c1j, c2j, c3j = solve_module(fvec, ud)

    pairs = [("c1", c1o, np.array(c1j)),
             ("c2", c2o, np.array(c2j)),
             ("c3", c3o, np.array(c3j))]

    case_max = 0.0
    for name, vo, vj in pairs:
        diff = np.abs(vo - vj)
        case_max = max(case_max, diff.max())
        print(f"  {name} orig : {fmt(vo)}")
        print(f"  {name} jax  : {fmt(vj)}")
        print(f"  {name} |Δ|  : {fmt(diff)}  max={diff.max():.2e}")
        print()

    max_errors.append((label, case_max))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  SUMMARY  (max |Δ| across all contact points, per test case)")
print(f"{'='*70}")
tol = 1e-4   # 0.1 mm tolerance
all_pass = True
for label, err in max_errors:
    status = "PASS" if err < tol else "FAIL"
    if err >= tol:
        all_pass = False
    print(f"  [{status}]  {err:.2e} m    {label}")

print(f"\n  Tolerance: {tol:.0e} m  ({'all pass' if all_pass else 'SOME FAILURES'})")