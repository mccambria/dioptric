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
from sc_b_field_of_magnetic import solve_B_from_odmr_order_invariant

import numpy as np

# ---------- NV geometry (crystal frame) ----------
def nv_axes():
    a = np.array([[ 1,  1,  1],
                  [-1,  1,  1],
                  [ 1, -1,  1],
                  [ 1,  1, -1]], float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a  # (4,3)

# Local NV frames: R_i rotates crystal-frame vectors into NV-local frame
def nv_local_frames():
    """Return list of 3x3 rotations R_i: v_local = R_i @ v_crystal."""
    nvs = nv_axes()
    frames = []
    for n in nvs:
        z = n / np.linalg.norm(n)
        # pick a helper to build an orthonormal basis
        helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z);  nx = np.linalg.norm(x)
        x = x/nx if nx>0 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        R = np.vstack([x, y, z])            # rows are local unit axes in crystal coords
        # We want v_local = R @ v_crystal
        frames.append(R)
    return frames  # list of (3,3)

# ---------- Spin-1 operators in local NV frame ----------
# Spin-1 matrices (same as before)
Sx = (1/np.sqrt(2))*np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], float)
Sy = (1/np.sqrt(2))*np.array([[0, -1j, 0],
                              [1j,  0, -1j],
                              [0,  1j,  0]], complex)
Sz = np.array([[ 1, 0, 0],
               [ 0, 0, 0],
               [ 0, 0,-1]], float)

# Basis kets in the NV-local Sz basis: |+1>, |0>, |-1>
ket_p1 = np.array([1.0, 0.0, 0.0], complex)
ket_0  = np.array([0.0, 1.0, 0.0], complex)
ket_m1 = np.array([0.0, 0.0, 1.0], complex)

def exact_f_minus_for_one_NV(B_crys_G, R_local, D_GHz=2.8785, E_MHz=0.0, gamma_e_MHz_per_G=2.8025):
    """
    Robust: choose ms~0 and ms~±1 by overlap with |0>,|±1> in the NV-local basis.
    Returns f_-(GHz) = E(lower of ±1-like) - E(0-like).
    """
    Bx, By, Bz = R_local @ np.asarray(B_crys_G, float).reshape(3)

    D_MHz = 1000.0*float(D_GHz)
    H_ZFS = D_MHz * (Sz @ Sz - (2/3)*np.eye(3))
    H_str = float(E_MHz) * (Sx @ Sx - Sy @ Sy)
    H_Zee = float(gamma_e_MHz_per_G) * (Bx*Sx + By*Sy + Bz*Sz)

    H = H_ZFS + H_str + H_Zee
    evals, evecs = np.linalg.eigh(H)   # columns of evecs are eigenvectors

    # Overlaps with Sz eigenkets in the *local* frame
    overlaps_0  = np.abs(evecs.conj().T @ ket_0 )**2   # weight of |0>
    overlaps_p1 = np.abs(evecs.conj().T @ ket_p1)**2   # weight of |+1>
    overlaps_m1 = np.abs(evecs.conj().T @ ket_m1)**2   # weight of |-1>

    idx_0  = int(np.argmax(overlaps_0))                # ms~0-like
    # For the ±1-like, forbid idx_0 and pick the most ±1-like among the rest
    candidates = [i for i in range(3) if i != idx_0]
    # score as total ±1 weight
    scores_pm1 = [(overlaps_p1[i] + overlaps_m1[i], i) for i in candidates]
    idx_pm1_hi = max(scores_pm1)[1]                    # most ±1-like among the two
    idx_pm1_lo = min(scores_pm1)[1]                    # the other ±1-like
    # lower of the two ±1-like energies:
    idx_minus = idx_pm1_hi if evals[idx_pm1_hi] < evals[idx_pm1_lo] else idx_pm1_lo

    f_minus_MHz = float(evals[idx_minus] - evals[idx_0])   # always ≥0 by construction
    return f_minus_MHz/1000.0


def exact_f_minus_quartet(B_crys_G, D_GHz=2.8785, E_MHz=0.0, gamma_e_MHz_per_G=2.8025):
    """Exact quartet (nv_axes order) using full diagonalization."""
    frames = nv_local_frames()
    return np.array([
        exact_f_minus_for_one_NV(B_crys_G, R, D_GHz, E_MHz, gamma_e_MHz_per_G)
        for R in frames
    ])

# ---- Increase |B| along current direction using the exact model ----
def plan_increase_Bmag_exact(f_current_GHz, D_GHz=2.8785, E_MHz=0.0, gamma_e_MHz_per_G=2.8025,
                             delta_Bmag_G=None, scale_Bmag=None):
    """
    1) Solve B0 from current quartet (your order-invariant solver).
    2) Set B1 = B̂0 * |B1| with |B1| = |B0|+delta or |B0|*scale.
    3) Predict new exact quartet with full Hamiltonian (E included).
    """
    # You already have this function; call your order-invariant solver to get B0 in crystal frame.
    # Below is a minimal drop-in call signature; replace with your existing function.
    from math import isclose

    # Reuse your existing order-invariant solver (not redefined here):
    # out0 = solve_B_from_odmr_order_invariant(f_current_GHz, D_GHz, gamma_e_MHz_per_G)
    # For clarity, assume it's in scope:
    out0 = solve_B_from_odmr_order_invariant(f_current_GHz, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    B0 = out0["B"]
    B0_mag = float(np.linalg.norm(B0))
    if B0_mag <= 0:
        raise ValueError("Solved |B| invalid.")

    if (delta_Bmag_G is None) == (scale_Bmag is None):
        raise ValueError("Provide exactly one of delta_Bmag_G or scale_Bmag.")

    B1_mag = B0_mag + float(delta_Bmag_G) if (delta_Bmag_G is not None) else B0_mag * float(scale_Bmag)
    if B1_mag <= 0:
        raise ValueError("Target |B| must be positive.")
    B_hat = B0 / B0_mag
    B1 = B_hat * B1_mag

    f0 = exact_f_minus_quartet(B0, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    f1 = exact_f_minus_quartet(B1, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    return {
        "B0_G": B0, "B1_G": B1, "B0_mag_G": B0_mag, "B1_mag_G": B1_mag,
        "f_before_nv_GHz": f0, "f_after_nv_GHz": f1,
    }

NV_LABELS = ["[1,1,1]","[-1,1,1]","[1,-1,1]","[1,1,-1]"]

def project_deltaB_to_spanK(K_3x2, dB):
    K = np.asarray(K_3x2, float).reshape(3,2)
    dB = np.asarray(dB, float).reshape(3)
    K_pinv = np.linalg.pinv(K)            # 2x3
    dB_proj = K @ (K_pinv @ dB)           # projection into Col(K)
    resid   = dB - dB_proj
    return dB_proj, resid

def solve_currents_ls(K_3x2, dB_proj, current_limits_A=None):
    K = np.asarray(K_3x2, float).reshape(3,2)
    dB_proj = np.asarray(dB_proj, float).reshape(3)
    I = np.linalg.lstsq(K, dB_proj, rcond=None)[0]     # [Iy, Iz]
    if current_limits_A is not None:
        lim = float(current_limits_A)
        I = np.clip(I, -lim, +lim)
    dB_ach = K @ I
    return I, dB_ach

def measured_baseline_in_nv_order(f_meas_GHz, D_GHz, gamma_e_MHz_per_G):
    f = np.asarray(f_meas_GHz, float).reshape(4)
    out = solve_B_from_odmr_order_invariant(f, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    perm = out["perm"]
    f_nv = np.empty(4, float)
    for j_in, i_nv in enumerate(perm):
        f_nv[i_nv] = f[j_in]
    return f_nv, out

def plan_to_target_Bmag_250G(f_current_GHz, D_GHz, gamma_e_MHz_per_G,
                             K_3x2=None, current_limits_A=None, E_MHz=0.0,
                             target_Bmag_G=250.0):
    # 1) solve current B0
    f_current_GHz = np.asarray(f_current_GHz, float).reshape(4)
    f_before_nv_meas, out0 = measured_baseline_in_nv_order(f_current_GHz, D_GHz, gamma_e_MHz_per_G)
    B0 = out0["B"]; B0_mag = float(np.linalg.norm(B0))
    if B0_mag <= 0: raise ValueError("Solved |B| is zero/invalid.")

    # 2) build target B1 = 250 G * Bhat
    Bhat = B0 / B0_mag
    B1_target = Bhat * float(target_Bmag_G)
    dB_target = B1_target - B0

    # 3) predict ideal quartet (exact model)
    f_before = exact_f_minus_quartet(B0, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    f_after_ideal = exact_f_minus_quartet(B1_target, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    result = {
        "B0_G": B0, "B0_mag_G": B0_mag,
        "B1_target_G": B1_target, "B1_target_mag_G": float(target_Bmag_G),
        "dB_target_G": dB_target,
        "f_before_nv_GHz_meas": f_before_nv_meas,      # your measured baseline mapped into nv_axes order
        "f_before_nv_GHz_model": f_before,             # model of baseline
        "f_after_ideal_nv_GHz": f_after_ideal
    }

    # 4) Coil-constrained plan (optional if K provided)
    if K_3x2 is not None:
        dB_proj, resid = project_deltaB_to_spanK(K_3x2, dB_target)
        I, dB_ach = solve_currents_ls(K_3x2, dB_proj, current_limits_A=current_limits_A)
        B1_ach = B0 + dB_ach
        f_after_ach = exact_f_minus_quartet(B1_ach, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
        result.update({
            "Iy_A": float(I[0]), "Iz_A": float(I[1]),
            "dB_projected_G": dB_proj, "dB_residual_unreachable_G": resid,
            "B1_achieved_G": B1_ach,
            "f_after_achieved_nv_GHz": f_after_ach
        })
    return result

def print_plan_250G(res, use_measured_before=True, title="Plan to |B|=250 G (crystal frame)"):
    labs = NV_LABELS
    f_before = res["f_before_nv_GHz_meas"] if use_measured_before else res["f_before_nv_GHz_model"]
    print(f"\n=== {title} ===")
    print(f"|B| now: {res['B0_mag_G']:.3f} G   →   target: {res['B1_target_mag_G']:.3f} G")
    print("B0 (G):        ", np.round(res["B0_G"], 6))
    print("B1_target (G): ", np.round(res["B1_target_G"], 6))
    print("ΔB_target (G): ", np.round(res["dB_target_G"], 6))

    print("\nIdeal (exact Hamiltonian):")
    print(" NV                f_- before (GHz)   f_- ideal (GHz)    Δf (MHz)")
    for i,lab in enumerate(labs):
        df = 1000*(res["f_after_ideal_nv_GHz"][i]-f_before[i])
        print(f" {lab:<12}   {f_before[i]:.9f}   {res['f_after_ideal_nv_GHz'][i]:.9f}   {df:+8.3f}")

    if "Iy_A" in res:
        print("\nCoil-constrained (projected into span(K)):")
        print(f" Iy (A): {res['Iy_A']:+.6f}   Iz (A): {res['Iz_A']:+.6f}")
        print(" ΔB_proj (G):    ", np.round(res["dB_projected_G"], 6))
        print(" ΔB_unreachable: ", np.round(res["dB_residual_unreachable_G"], 6), "(this part your coils cannot make)")
        print(" B1_achieved (G):", np.round(res["B1_achieved_G"], 6))
        print("\n NV                f_- before (GHz)   f_- achieved (GHz)  Δf (MHz)")
        for i,lab in enumerate(labs):
            fa = res["f_after_achieved_nv_GHz"][i]
            df = 1000*(fa - f_before[i])
            print(f" {lab:<12}   {f_before[i]:.9f}   {fa:.9f}   {df:+8.3f}")

# ----------------- Example -----------------
if __name__ == "__main__":
    # try any ordering you like:

    # --- use the exact-model helpers you pasted earlier ---
    # nv_axes(), nv_local_frames(), exact_f_minus_for_one_NV(), exact_f_minus_quartet()
    # solve_B_from_odmr_order_invariant(...)
    # plan_increase_Bmag_exact(...)      # from the code I gave
    # (optional) your print helper print_Bmag_increase_plan(...)

    # Your baseline quartet (any order) and constants
    f_base = [2.7660, 2.7851, 2.8235, 2.8405]   # GHz, measured
    D      = 2.8785                              # GHz
    gamma  = 2.8025                              # MHz/G

    # =========================
    # EXAMPLE A: Just predict
    # Increase |B| by +5 G, keep direction fixed; predict new quartet (exact model, E=0)
    # =========================
    plan = plan_increase_Bmag_exact(
        f_current_GHz=f_base,
        D_GHz=D,
        E_MHz=0.0,                 # set your strain here if known
        gamma_e_MHz_per_G=gamma,
        delta_Bmag_G=150.0           # OR use scale_Bmag=1.10 for +10%
    )
    print("\n=== Increase |B| by +5 G (exact model) ===")
    print(f"|B|: {plan['B0_mag_G']:.3f} G  →  {plan['B1_mag_G']:.3f} G")
    print("B0 (G):", np.round(plan["B0_G"], 6))
    print("B1 (G):", np.round(plan["B1_G"], 6))
    NV_LABELS = ["[1,1,1]","[-1,1,1]","[1,-1,1]","[1,1,-1]"]
    print("\nNV (nv_axes order)      f_- before (GHz)   f_- after (GHz)    Δf (MHz)")
    for i, lab in enumerate(NV_LABELS):
        df = 1000*(plan["f_after_nv_GHz"][i] - plan["f_before_nv_GHz"][i])
        print(f" {lab:<10}          {plan['f_before_nv_GHz'][i]:.9f}    {plan['f_after_nv_GHz'][i]:.9f}    {df:+8.3f}")

    # =========================
    # EXAMPLE B: Also suggest Iy/Iz to realize ΔB
    # =========================
    # If you already have a coil map K_crys (3x2, columns are dB/dIy and dB/dIz in crystal axes), use it here.
    # Start with a rough diagonal (replace with your fitted K when you have it):
    K_diag = np.array([
        [ 0.0,   0.0   ],   # dBx/dIy, dBx/dIz
        [-6.37,  0.0   ],   # dBy/dIy, dBy/dIz
        [ 0.0,  -17.39 ],   # dBz/dIy, dBz/dIz
    ])  # G/A  (placeholder!)

    # Recompute plan so we have ΔB = B1 - B0, then solve least-squares for currents
    plan2 = plan_increase_Bmag_exact(
        f_current_GHz=f_base, D_GHz=D, E_MHz=0.0, gamma_e_MHz_per_G=gamma,
        scale_Bmag=3.10   # example: +10% magnitude
    )
    B0, B1 = plan2["B0_G"], plan2["B1_G"]
    dB = B1 - B0

    # Solve min-norm currents: minimize ||K dI - dB||_2
    K = K_diag
    lhs = K.T @ K
    rhs = K.T @ dB
    dI = np.linalg.lstsq(lhs, rhs, rcond=None)[0]   # [Iy, Iz]
    Iy, Iz = float(dI[0]), float(dI[1])

    # Check what ΔB you actually get with that K, and the residual
    dB_ach = K @ dI
    res    = dB_ach - dB

    print("\n=== Coil suggestion for +10% |B| (LS, crystal frame) ===")
    print(f"Iy (A): {Iy:+.6f}   Iz (A): {Iz:+.6f}")
    print("Target ΔB (G):    ", np.round(dB, 6))
    print("Achieved ΔB (G):  ", np.round(dB_ach, 6))
    print("Residual ΔB (G):  ", np.round(res, 6))

    # Predict quartet after applying achieved ΔB (exact model)
    f_after_nv = exact_f_minus_quartet(B0 + dB_ach, D_GHz=D, E_MHz=0.0, gamma_e_MHz_per_G=gamma)
    print("\nNV (nv_axes order)      f_- before (GHz)   f_- after (GHz)    Δf (MHz)")
    for i, lab in enumerate(NV_LABELS):
        f0 = plan2["f_before_nv_GHz"][i]
        df = 1000*(f_after_nv[i] - f0)
        print(f" {lab:<10}          {f0:.9f}    {f_after_nv[i]:.9f}    {df:+8.3f}")

    # =========================
    # (Optional) If you want to incorporate measured cross-talk:
    # Build K from two one-coil shots (quartets at (Iy,0) and (0,Iz)) and ref (0,0),
    # solve B for each, then finite-difference:
    #   dB/dIy ≈ (B(Iy,0)-B(0,0))/Iy
    #   dB/dIz ≈ (B(0,Iz)-B(0,0))/Iz
    # Put those as columns of K and re-run the LS step above.
    # =========================

    # # Your measured baseline quartet (any order)
    # f_base = [2.7660, 2.7851, 2.8235, 2.8405]   # GHz
    # D = 2.8785
    # gamma = 2.8025
    # E = 0.0  # MHz (set if you have a measured strain splitting)

    # # Replace this with your fitted coil map K_crys (3x2, columns = dB/dIy and dB/dIz in crystal axes)
    # # Placeholder (y-only affects By, z-only affects Bz):
    # K_crys = np.array([
    #     [ 0.0,   0.0],     # dBx/dIy, dBx/dIz
    #     [-6.37,  0.0],     # dBy/dIy, dBy/dIz   (example from your one-shot estimate)
    #     [ 0.0, -17.39],    # dBz/dIy, dBz/dIz
    # ])  # G/A

    # res = plan_to_target_Bmag_250G(
    #     f_current_GHz=f_base,
    #     D_GHz=D,
    #     gamma_e_MHz_per_G=gamma,
    #     K_3x2=K_crys,
    #     current_limits_A=14.0,     # clip if you want
    #     E_MHz=E,
    #     target_Bmag_G=200.0
    # )
    # print_plan_250G(res, use_measured_before=True)
