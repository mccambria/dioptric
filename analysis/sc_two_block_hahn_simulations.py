# Bloch-sphere visualization for a two-block Hahn sequence (conceptual)
# - π/2_x – τ – π_x – τ – π/2_±x   |   lag Δ   |   π/2_x – τ – π_x – τ – π/2_±x
# We show the state vector path on the Bloch sphere for one run, given
# phase accumulations φ1 and φ2 during the two Hahn blocks.
#
# Notes for this demo:
# - We model free evolution as rotation about z by angle φ (accumulated during τ…π…τ).
# - π/2 and π pulses are instantaneous rotations about x.
# - We then draw the two blocks separated by a gap (Δ not explicitly drawn on the sphere).
#
# You can tweak phi1, phi2 to see how the lag-dependent phase shows up in the final vector.

import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

kpl.init_kplotlib()
# ---- Helpers for rotations and drawing ----
def Rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    # SU(2) mapping on Bloch vector => equivalent SO(3) rotation matrix about x by angle theta
    return np.array([[1,      0,       0],
                     [0,  np.cos(theta), -np.sin(theta)],
                     [0,  np.sin(theta),  np.cos(theta)]])

def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0],
                     [ np.sin(theta),  np.cos(theta), 0],
                     [ 0,              0,             1]])

def path_segment(Rlist, v0):
    pts = [v0]
    v = v0.copy()
    for R in Rlist:
        v = R @ v
        pts.append(v)
    return np.stack(pts, axis=0)

def draw_bloch(ax, radius=1.0, nmer=32, npar=16):
    # Sphere
    u = np.linspace(0, 2*np.pi, nmer)
    v = np.linspace(0, np.pi, npar)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.5)
    # axes
    ax.plot([-radius, radius], [0,0], [0,0], linewidth=1.0)  # x
    ax.plot([0,0], [-radius, radius], [0,0], linewidth=1.0)  # y
    ax.plot([0,0], [0,0], [-radius, radius], linewidth=1.0)  # z
    ax.text(1.05*radius,0,0,'x')
    ax.text(0,1.05*radius,0,'y')
    ax.text(0,0,1.05*radius,'z')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def plot_block(ax, vstart, phi, final_half_pi_sign=+1, label_prefix="Block"):
    """
    Simulate one Hahn block:
      π/2_x  -> free (φ/2 about z) -> π_x -> free (φ/2 about z) -> π/2_x (or -π/2_x)
    Returns the final Bloch vector.
    """
    segs = []
    v = vstart.copy()
    # π/2 about +x
    R = Rx(np.pi/2)
    v = R @ v; segs.append(v.copy())
    # free evolution φ/2 about z
    R = Rz(phi/2.0); v = R @ v; segs.append(v.copy())
    # π about +x
    R = Rx(np.pi); v = R @ v; segs.append(v.copy())
    # free evolution φ/2 about z
    R = Rz(phi/2.0); v = R @ v; segs.append(v.copy())
    # final ±π/2 about x
    R = Rx(final_half_pi_sign*np.pi/2); v = R @ v; segs.append(v.copy())

    # Plot the piecewise path
    pts = np.vstack(([vstart], np.array(segs)))
    ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2.0)
    # annotate start and end
    ax.scatter([vstart[0]], [vstart[1]], [vstart[2]], s=30)
    ax.scatter([v[0]], [v[1]], [v[2]], s=30)
    return v

# ---- Choose example phases for two blocks (tune these) ----
# Example: narrowband field at 2.18 MHz, pick a lag so φ2 = φ1 + δ
phi1 = 0.7          # phase accumulated during block 1
phi2 = -0.4         # phase accumulated during block 2 (can be function of Δ)

# Orientation sign s = ±1 can be absorbed into phi by phi -> s*phi. We'll show +1 here.
s = +1
phi1 *= s; phi2 *= s

# ---- Build figure ----
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
draw_bloch(ax)

# Start at |0> = +z
v0 = np.array([0.0, 0.0, 1.0])

# Block 1
v1 = plot_block(ax, v0, phi=phi1, final_half_pi_sign=+1, label_prefix="Block 1")

# (Lag Δ happens here; we don't evolve on the sphere for visualization simplicity.)

# Block 2 starting from v1
v2 = plot_block(ax, v1, phi=phi2, final_half_pi_sign=+1, label_prefix="Block 2")

ax.set_title("Two-block Hahn on the Bloch sphere\n(π/2–τ–π–τ–π/2 per block; phases φ₁, φ₂ about z)")


# Bloch-sphere visualization for a *single* Hahn block, as a separate figure.
# Reuses the same helpers and style as the two-block plot you liked.

def draw_bloch(ax, radius=1.0, nmer=32, npar=16):
    u = np.linspace(0, 2*np.pi, nmer)
    v = np.linspace(0, np.pi, npar)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.5)
    ax.plot([-radius, radius], [0,0], [0,0], linewidth=1.0)  # x
    ax.plot([0,0], [-radius, radius], [0,0], linewidth=1.0)  # y
    ax.plot([0,0], [0,0], [-radius, radius], linewidth=1.0)  # z
    ax.text(1.05*radius,0,0,'x')
    ax.text(0,1.05*radius,0,'y')
    ax.text(0,0,1.05*radius,'z')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def plot_single_hahn_block(ax, phi, final_half_pi_sign=+1):
    """
    π/2_x  -> free (φ/2 about z) -> π_x -> free (φ/2 about z) -> π/2_x (or -π/2_x)
    Returns start and end vectors for convenience.
    """
    v = np.array([0.0, 0.0, 1.0])  # start at |0> = +z
    pts = [v.copy()]

    # π/2_x
    v = Rx(np.pi/2) @ v; pts.append(v.copy())
    # free φ/2
    v = Rz(phi/2.0) @ v; pts.append(v.copy())
    # π_x
    v = Rx(np.pi) @ v; pts.append(v.copy())
    # free φ/2
    v = Rz(phi/2.0) @ v; pts.append(v.copy())
    # final ±π/2_x
    v = Rx(final_half_pi_sign*np.pi/2) @ v; pts.append(v.copy())

    pts = np.stack(pts, axis=0)
    ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2.0)
    ax.scatter([pts[0,0]], [pts[0,1]], [pts[0,2]], s=30)  # start
    ax.scatter([pts[-1,0]], [pts[-1,1]], [pts[-1,2]], s=30)  # end
    return pts[0], pts[-1]

# ---- Choose example phase for the single block ----
phi = 0.8    # adjust to taste; this is the net phase accumulated inside the Hahn block

# ---- Build the figure ----
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
draw_bloch(ax)

plot_single_hahn_block(ax, phi=phi, final_half_pi_sign=+1)

ax.set_title("Single Hahn block on the Bloch sphere\n(π/2–τ–π–τ–π/2; net phase φ about z)")


kpl.show(block=True)