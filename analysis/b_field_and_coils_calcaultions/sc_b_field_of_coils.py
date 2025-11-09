# -*- coding: utf-8 -*-
"""
Extract the magnetic field (crystallographic axes only)
Created on March 23th, 2025
@author: Saroj Chand
"""

import numpy as np
from itertools import product, permutations

import numpy as np
from itertools import permutations, product
from sc_b_field_calculations import solve_B_from_odmr_order_invariant

import numpy as np
# validate_coils.py
# Reconstruct coil shots, predict exact quartets, and compute residuals


# ---------- Constants ----------
D_GHz   = 2.8785
gamma   = 2.8025   # MHz/G
E_MHz   = 0.0

NV_LABELS = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]

# ===== Paste your fitted values here =====
B0 = np.array([-46.27557688, -17.16599864, -5.70139829], float)    # baseline field (G)
Ky = np.array([  6.310500,   0.030810, -1.638782], float)    # dB/dIy (G/A)
Kz = np.array([  6.468884,   4.104149, -3.782627], float)    # dB/dIz (G/A)
# ========================================

# ---------- NV geometry ----------
def nv_axes():
    a = np.array([[ 1,  1,  1],
                  [-1,  1,  1],
                  [ 1, -1,  1],
                  [ 1,  1, -1]], float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a  # (4,3)

# Build a rotation that maps crystal-> NV-local frame (rows = local axes)
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

# ---------- Exact S=1 Hamiltonian and f_- ----------
Sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]], complex)
Sy = (1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], complex)
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], float)

ket_p1 = np.array([1,0,0], complex)
ket_0  = np.array([0,1,0], complex)
ket_m1 = np.array([0,0,1], complex)

def exact_f_minus_for_one_NV(B_crys_G, R_local, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma):
    Bx, By, Bz = R_local @ np.asarray(B_crys_G, float).reshape(3)
    D_MHz = 1000.0*D_GHz
    H = D_MHz*(Sz@Sz - (2/3)*np.eye(3)) + E_MHz*(Sx@Sx - Sy@Sy) + gamma_e_MHz_per_G*(Bx*Sx + By*Sy + Bz*Sz)
    evals, evecs = np.linalg.eigh(H)

    # Identify |0>-like and the lower of the two |±1>-like
    overlaps_0  = np.abs(evecs.conj().T @ ket_0 )**2
    overlaps_p1 = np.abs(evecs.conj().T @ ket_p1)**2
    overlaps_m1 = np.abs(evecs.conj().T @ ket_m1)**2
    i0  = int(np.argmax(overlaps_0))
    cand = [i for i in range(3) if i != i0]
    scores = [(overlaps_p1[i] + overlaps_m1[i], i) for i in cand]
    ih = max(scores)[1]; il = min(scores)[1]
    i_minus = ih if evals[ih] < evals[il] else il
    return float(evals[i_minus] - evals[i0]) / 1000.0  # GHz

def exact_f_minus_quartet(B_crys_G, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma):
    frames = nv_local_frames()
    return np.array([exact_f_minus_for_one_NV(B_crys_G, R, D_GHz, E_MHz, gamma_e_MHz_per_G)
                     for R in frames])

# ---------- Utilities ----------
def B_from_currents(Iy, Iz):
    return B0 + Ky*Iy + Kz*Iz

def residuals_MHz(f_meas, f_pred):
    return 1000.0*(np.asarray(f_pred)-np.asarray(f_meas))

def print_shot_report(name, Iy, Iz, f_meas_any_order=None):
    """
    If f_meas_any_order is provided (length-4 list/array), we print both predicted
    quartet (nv order) and the measured (sorted into nv order via best assignment).
    If only currents are given, we just print the predicted.
    """
    B = B_from_currents(Iy, Iz)
    f_pred = exact_f_minus_quartet(B, D_GHz, E_MHz, gamma)

    print(f"\n=== Shot: {name}  (Iy={Iy:.3f} A, Iz={Iz:.3f} A) ===")
    print("B (G):", np.round(B, 6))
    print("Predicted quartet (nv_axes order):")
    for i,lab in enumerate(NV_LABELS):
        print(f"  {lab:<10}  f_- = {f_pred[i]:.9f} GHz")

    if f_meas_any_order is None:
        return

    # Map measured (any order) into nv order by minimizing |n·B| mismatch
    f_meas = np.asarray(f_meas_any_order, float).reshape(4)
    n = nv_axes()
    abs_pred = np.abs(n @ B)                      # |n·B| (G)
    abs_meas = (D_GHz - f_meas) * 1000.0 / gamma # |n·B| from meas
    # find best permutation
    from itertools import permutations
    best_perm, best_err = None, np.inf
    for perm in permutations(range(4)):
        err = np.sum((abs_meas[list(range(4))] - abs_pred[list(perm)])**2)
        if err < best_err:
            best_err, best_perm = err, perm
    f_meas_nv = np.empty(4, float)
    for j,i in enumerate(best_perm):
        f_meas_nv[i] = f_meas[j]

    # Residuals
    r = residuals_MHz(f_meas_nv, f_pred)
    print("\nMeasured quartet (mapped into nv_axes order):")
    for i,lab in enumerate(NV_LABELS):
        print(f"  {lab:<10}  f_meas = {f_meas_nv[i]:.9f} GHz   f_pred = {f_pred[i]:.9f} GHz   Δf = {r[i]:+8.3f} MHz")
    print(f"RMS(Δf) = {np.sqrt(np.mean(r**2)):.3f} MHz")

# ---------- Batch validator ----------
def validate_all(shots):
    """
    shots: list of dicts, each with:
      name, Iy, Iz, f_meas (optional 4-list in any order)
    """
    for s in shots:
        print_shot_report(s["name"], s["Iy"], s["Iz"], s.get("f_meas"))

# ---------- Optional: plan to increase |B| by ΔG along current direction ----------
def plan_increase_Bmag(delta_Bmag_G):
    Bmag0 = float(np.linalg.norm(B0))
    if Bmag0 <= 0: raise ValueError("Invalid |B0|")
    Bhat  = B0 / Bmag0
    B1    = B0 + Bhat*delta_Bmag_G
    dB    = B1 - B0

    # Least-squares currents to realize dB with K = [Ky Kz]
    K = np.column_stack([Ky, Kz])  # 3x2
    # Solve min ||K dI - dB||_2
    dI, *_ = np.linalg.lstsq(K, dB, rcond=None)
    Iy, Iz = float(dI[0]), float(dI[1])
    B_ach  = B_from_currents(Iy, Iz)
    return {
        "delta_Bmag_G": delta_Bmag_G,
        "Iy_A": Iy, "Iz_A": Iz,
        "B_target_G": B1, "B_achieved_G": B_ach,
        "f_target_GHz": exact_f_minus_quartet(B1, D_GHz, E_MHz, gamma),
        "f_achieved_GHz": exact_f_minus_quartet(B_ach, D_GHz, E_MHz, gamma),
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    # Put your measured quartets here (any order inside each list):
    shots = [
        {"name":"(0,0)",   "Iy":0.0, "Iz":0.0, "f_meas":[2.7666, 2.7851, 2.8222, 2.8406]},
        {"name":"(1,0)",   "Iy":1.0, "Iz":0.0, "f_meas":[2.7457, 2.7988, 2.8344, 2.8765]},
        {"name":"(0,1)",   "Iy":0.0, "Iz":1.0, "f_meas":[2.7694, 2.8403, 2.7991, 2.8406]},
        {"name":"(1,1)",   "Iy":1.0, "Iz":1.0, "f_meas":[2.7457, 2.8170, 2.8100, 2.8751]},
        # You can add a validation point like (1,-1), etc.
        # {"name":"(1,-1)", "Iy":1.0, "Iz":-1.0, "f_meas":[...]},
    ]
    validate_all(shots)

    # Example: ask for +10 G along current direction and see coils & predicted peaks
    plan = plan_increase_Bmag(delta_Bmag_G=10.0)
    print("\n=== Plan: increase |B| by +10 G along current direction ===")
    print(f"Iy = {plan['Iy_A']:+.4f} A,  Iz = {plan['Iz_A']:+.4f} A")
    print("B target (G):   ", np.round(plan["B_target_G"], 6))
    print("B achieved (G): ", np.round(plan["B_achieved_G"], 6))
    print("\nNV (nv_axes order)    f_- target (GHz)   f_- achieved (GHz)   Δf (MHz)")
    for i,lab in enumerate(NV_LABELS):
        df = 1000.0*(plan["f_achieved_GHz"][i] - plan["f_target_GHz"][i])
        print(f" {lab:<12}    {plan['f_target_GHz'][i]:.9f}      {plan['f_achieved_GHz'][i]:.9f}     {df:+8.3f}")
