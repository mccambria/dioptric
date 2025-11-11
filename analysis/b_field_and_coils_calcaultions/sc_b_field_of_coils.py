# -*- coding: utf-8 -*-
"""
Extract the magnetic field (crystallographic axes only)
Created on March 23th, 2025
@author: Saroj Chand
"""
# -*- coding: utf-8 -*-
"""
validate_coils.py
Reconstruct coil shots, predict exact quartets, and compute residuals
(uses new B0 and K you solved from four ODMR quartets)
"""

import numpy as np
from itertools import permutations

# ---------- Constants ----------
D_GHz   = 2.8785
GAMMA   = 2.8025   # MHz/G
E_MHz   = 0.0

NV_LABELS = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]

# ---------- New calibration (crystal frame) ----------
# Baseline field from (0,0)
B0 = np.array([-46.275577, -17.165999,  -5.701398], float)  # Gauss

# Coil→Field map K (columns = dB/dI_ch1, dB/dI_ch2), Gauss per Amp
K = np.column_stack([
    [ +0.803449,  -4.758891, -15.327336],   # dB/dI_ch1
    [ -0.741645,  +4.202657,  -4.635283],   # dB/dI_ch2
]).astype(float)

# Optional: common drift vector that applied to the old coil-on runs only
DRIFT_OLD = np.array([ +6.134025, -4.403519, +5.701398 ], float)  # Gauss

# ---------- NV geometry ----------
def nv_axes():
    a = np.array([[ 1,  1,  1],
                  [-1,  1,  1],
                  [ 1, -1,  1],
                  [ 1,  1, -1]], float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a  # (4,3)

def nv_local_frames():
    frames = []
    for n in nv_axes():
        z = n / np.linalg.norm(n)
        helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z); nx = np.linalg.norm(x)
        x = x/nx if nx>0 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        frames.append(np.vstack([x, y, z]))
    return frames

# ---------- Exact S=1 Hamiltonian (f_- prediction) ----------
Sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]], complex)
Sy = (1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], complex)
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], float)

ket_p1 = np.array([1,0,0], complex)
ket_0  = np.array([0,1,0], complex)
ket_m1 = np.array([0,0,1], complex)

def exact_f_minus_for_one_NV(B_crys_G, R_local, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=GAMMA):
    Bx, By, Bz = R_local @ np.asarray(B_crys_G, float).reshape(3)
    H = (1000.0*D_GHz)*(Sz@Sz - (2/3)*np.eye(3)) \
        + E_MHz*(Sx@Sx - Sy@Sy) \
        + gamma_e_MHz_per_G*(Bx*Sx + By*Sy + Bz*Sz)
    evals, evecs = np.linalg.eigh(H)

    overlaps_0  = np.abs(evecs.conj().T @ ket_0 )**2
    overlaps_p1 = np.abs(evecs.conj().T @ ket_p1)**2
    overlaps_m1 = np.abs(evecs.conj().T @ ket_m1)**2
    i0  = int(np.argmax(overlaps_0))
    cand = [i for i in range(3) if i != i0]
    scores = [(overlaps_p1[i] + overlaps_m1[i], i) for i in cand]
    ih = max(scores)[1]; il = min(scores)[1]
    i_minus = ih if evals[ih] < evals[il] else il
    return float(evals[i_minus] - evals[i0]) / 1000.0  # GHz

def exact_f_minus_quartet(B_crys_G, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=GAMMA):
    frames = nv_local_frames()
    return np.array([exact_f_minus_for_one_NV(B_crys_G, R, D_GHz, E_MHz, gamma_e_MHz_per_G)
                     for R in frames])

# ---------- Utilities ----------
def B_from_currents(I_ch1, I_ch2, drift=None):
    """Field for given currents (A). Optionally add a drift vector (Gauss)."""
    I = np.array([float(I_ch1), float(I_ch2)], float)
    B = B0 + K @ I
    if drift is not None:
        B = B + np.asarray(drift, float).reshape(3)
    return B

def residuals_MHz(f_meas, f_pred):
    return 1000.0*(np.asarray(f_pred) - np.asarray(f_meas))

def assign_measured_to_nv_order(f_meas_GHz, B_crys_G):
    """
    Map 4 measured peaks (any order) into NV-axes order by matching |n·B|.
    Returns array in NV order.
    """
    f_meas = np.asarray(f_meas_GHz, float).reshape(4)
    n = nv_axes()
    abs_pred = np.abs(n @ B_crys_G)                      # Gauss
    abs_meas = (D_GHz - f_meas) * 1000.0 / GAMMA        # Gauss

    best_perm, best_err = None, np.inf
    idx = np.arange(4)
    for perm in permutations(range(4)):
        err = np.sum((abs_meas[idx] - abs_pred[list(perm)])**2)
        if err < best_err:
            best_err, best_perm = err, perm
    f_nv = np.empty(4, float)
    for j, i in enumerate(best_perm):
        f_nv[i] = f_meas[j]
    return f_nv

def print_shot_report(name, I_ch1, I_ch2, f_meas_any_order=None, drift=None):
    B = B_from_currents(I_ch1, I_ch2, drift=drift)
    f_pred = exact_f_minus_quartet(B, D_GHz, E_MHz, GAMMA)

    print(f"\n=== Shot: {name}  (I_ch1={I_ch1:.3f} A, I_ch2={I_ch2:.3f} A) ===")
    print("B (G):   ", np.round(B, 6))
    print("|B| (G): ", f"{np.linalg.norm(B):.6f}")
    print("Predicted quartet (NV-axes order):")
    for i, lab in enumerate(NV_LABELS):
        print(f"  {lab:<10}  f_- = {f_pred[i]:.9f} GHz")

    if f_meas_any_order is None:
        return

    f_meas_nv = assign_measured_to_nv_order(f_meas_any_order, B)
    r = residuals_MHz(f_meas_nv, f_pred)
    print("\nMeasured quartet (mapped to NV-axes order):")
    for i, lab in enumerate(NV_LABELS):
        print(f"  {lab:<10}  f_meas = {f_meas_nv[i]:.9f} GHz   "
              f"f_pred = {f_pred[i]:.9f} GHz   Δf = {r[i]:+8.3f} MHz")
    print(f"RMS(Δf) = {np.sqrt(np.mean(r**2)):.3f} MHz")

# ---------- Batch validator ----------
def validate_all(shots, drift=None):
    """
    shots: list of dicts with
      - name (str)
      - I_ch1 (float)
      - I_ch2 (float)
      - f_meas (optional 4-list in any order)
    drift: optional 3-vector (Gauss) to add for all shots (use DRIFT_OLD for that session)
    """
    for s in shots:
        print_shot_report(s["name"], s["I_ch1"], s["I_ch2"], s.get("f_meas"), drift=drift)

# ---------- Plan: increase |B| by a given number of Gauss along current B̂ ----------
def plan_increase_Bmag(delta_Bmag_G, use_drift=None):
    B_start = B_from_currents(0.0, 0.0, drift=use_drift)
    Bmag0 = float(np.linalg.norm(B_start))
    if Bmag0 <= 0:
        raise ValueError("Invalid |B| at start.")
    Bhat  = B_start / Bmag0
    B_goal = B_start + Bhat*float(delta_Bmag_G)
    dB_req = B_goal - B0 - (np.asarray(use_drift) if use_drift is not None else 0.0)

    # Least-squares currents for dB_req in span(K)
    dI, *_ = np.linalg.lstsq(K, dB_req, rcond=None)
    I1, I2 = float(dI[0]), float(dI[1])
    B_ach  = B_from_currents(I1, I2, drift=use_drift)

    return {
        "delta_Bmag_G": float(delta_Bmag_G),
        "I_ch1_A": I1, "I_ch2_A": I2,
        "B_start_G": B_start, "B_goal_G": B_goal, "B_achieved_G": B_ach,
        "f_goal_GHz": exact_f_minus_quartet(B_goal, D_GHz, E_MHz, GAMMA),
        "f_ach_GHz":  exact_f_minus_quartet(B_ach,  D_GHz, E_MHz, GAMMA),
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    # Measured quartets (GHz), any internal order is OK
    shots = [
        {"name":"(0,0)", "I_ch1":0.0, "I_ch2":0.0, "f_meas":[2.7666, 2.7851, 2.8222, 2.8406]},
        {"name":"(1,0)", "I_ch1":1.0, "I_ch2":0.0, "f_meas":[2.7457, 2.7988, 2.8344, 2.8765]},
        {"name":"(0,1)", "I_ch1":0.0, "I_ch2":1.0, "f_meas":[2.7694, 2.8403, 2.7991, 2.8406]},
        {"name":"(1,1)", "I_ch1":1.0, "I_ch2":1.0, "f_meas":[2.7457, 2.8170, 2.8100, 2.8751]},
    ]

    # Validate without drift for new runs:
    validate_all(shots, drift=None)

    # If you want to “correct” the historical session that had a common drift, use:
    # validate_all(shots, drift=DRIFT_OLD)

    # Example plan: +10 G along current field direction (no drift)
    plan = plan_increase_Bmag(delta_Bmag_G=10.0, use_drift=None)
    print("\n=== Plan: increase |B| by +10 G along current direction ===")
    print(f"I_ch1 = {plan['I_ch1_A']:+.4f} A,  I_ch2 = {plan['I_ch2_A']:+.4f} A")
    print("B start (G):    ", np.round(plan["B_start_G"], 6), f"  |B|={np.linalg.norm(plan['B_start_G']):.6f} G")
    print("B goal  (G):    ", np.round(plan["B_goal_G"],  6), f"  |B|={np.linalg.norm(plan['B_goal_G']):.6f} G")
    print("B achieved (G): ", np.round(plan["B_achieved_G"], 6), f"  |B|={np.linalg.norm(plan['B_achieved_G']):.6f} G")
    print("\nNV (NV-axes order)   f_- goal (GHz)     f_- achieved (GHz)   Δf (MHz)")
    for i, lab in enumerate(NV_LABELS):
        df = 1000.0*(plan["f_ach_GHz"][i] - plan["f_goal_GHz"][i])
        print(f" {lab:<12}   {plan['f_goal_GHz'][i]:.9f}      {plan['f_ach_GHz'][i]:.9f}     {df:+8.3f}")
