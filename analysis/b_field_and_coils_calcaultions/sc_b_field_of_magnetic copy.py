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

# ---------- Geometry ----------
def nv_axes():
    a = np.array([[ 1,  1,  1],
                  [-1,  1,  1],
                  [ 1, -1,  1],
                  [ 1,  1, -1]], float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a  # (4,3)

NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]

# ---------- Forward model ----------
def f_minus_from_B(B_vec_G, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    n = nv_axes()
    abs_proj_G = np.abs(n @ np.asarray(B_vec_G, float).reshape(3))
    return D_GHz - (gamma_e_MHz_per_G * abs_proj_G) / 1000.0  # (4,)

# ---------- Solvers you already have (order-invariant) ----------
def solve_B_from_odmr_order_invariant(
    f_ms_minus_GHz, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025,
    equal_tol=1e-12, round_decimals=12
):
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)
    n = nv_axes()
    abs_nB = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    best_res = np.inf
    cand = None
    for perm in permutations(range(4)):
        b_mag = abs_nB[list(perm)]
        for s in product([-1.0, +1.0], repeat=4):
            s = np.asarray(s, float)
            rhs = s * b_mag
            B, *_ = np.linalg.lstsq(n, rhs, rcond=None)
            proj = n @ B
            if not np.all(np.sign(proj + 1e-12) == np.sign(rhs + 1e-12)):
                continue
            rvec = proj - rhs
            res = float(np.linalg.norm(rvec))
            if res < best_res - equal_tol or (
                abs(res - best_res) <= equal_tol and tuple(np.round(B, round_decimals)) < tuple(np.round(cand[1], round_decimals))  # tie-break
            ):
                best_res, cand = res, (res, B, proj, s, b_mag, perm)

    if cand is None:
        raise RuntimeError("No consistent solution; check inputs.")
    _, B, proj, s, b_mag, perm = cand
    B_mag = float(np.linalg.norm(B))
    return {
        "B": B,
        "B_mag": B_mag,
        "projections": proj,
        "signs": s,
        "b_magnitudes": b_mag,
        "perm": perm,
        "f_minus_nvaxes_GHz": f_minus_from_B(B, D_GHz, gamma_e_MHz_per_G),
    }

# ---------- Core: compute currents to hit target NV positions ----------
def currents_to_hit_target_f(
    f_current_GHz,               # 4 f_- lines, any order (current)
    f_target_by_idx,             # dict {ori_idx: target_f_- (GHz)}; ori_idx in nv_axes order 0..3
    K_crys_3x2,                  # 3x2 coil map in crystal axes (G/A): [dB/dIy, dB/dIz]
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    weights_by_idx=None,         # optional dict {ori_idx: weight}, default 1.0
    coil_limit_A=None            # optional scalar max |Iy|,|Iz| for safety clamp
):
    """
    Linearized LS solution for Iy, Iz to move chosen NV lines to target f_-.
    Works in the crystal frame. Returns currents and predicted results.
    """
    n = nv_axes()
    out = solve_B_from_odmr_order_invariant(f_current_GHz, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    B0 = out["B"]
    proj0 = n @ B0                           # n·B0 (signed)
    sgn = np.sign(proj0 + 1e-18)             # baseline signs

    # Rows for selected constraints
    sel = sorted(f_target_by_idx.keys())
    m = len(sel)
    if m == 0:
        raise ValueError("Provide at least one target NV.")

    # Build A * dI ≈ b
    # f_- = D - gamma*|n·B|/1000 ⇒ Δ|n·B| (G) = -(1000/gamma) Δf_- (MHz)
    # Linearize: Δ|n·B| ≈ s_i (n_i · ΔB) = s_i n_i^T K dI
    K = np.asarray(K_crys_3x2, float).reshape(3,2)
    A = np.zeros((m, 2), float)
    b = np.zeros((m,), float)
    w = np.ones((m,), float)

    # current f_- in nv_axes order (for reference)
    f0_nv = f_minus_from_B(B0, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    for row, i in enumerate(sel):
        ni = n[i]
        Ai = sgn[i] * (ni @ K)                  # shape (2,)
        A[row, :] = Ai
        f0_i = f0_nv[i]
        df_i_MHz = (f_target_by_idx[i] - f0_i)
        b[row] = -(1000.0 / gamma_e_MHz_per_G) * df_i_MHz
        if weights_by_idx and (i in weights_by_idx):
            w[row] = float(weights_by_idx[i])

    # Weighted least squares
    W = np.diag(w)
    lhs = A.T @ W @ A
    rhs = A.T @ W @ b
    dI = np.linalg.lstsq(lhs, rhs, rcond=None)[0]  # (Iy, Iz)

    # Optional clamp
    if coil_limit_A is not None:
        dI = np.clip(dI, -abs(coil_limit_A), +abs(coil_limit_A))

    # Predict new B and new quartet
    dB = K @ dI
    B_new = B0 + dB
    f_new_nv = f_minus_from_B(B_new, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    # Residuals on constrained axes (in MHz)
    res_MHz = {}
    for row, i in enumerate(sel):
        res_MHz[i] = 1000.0 * (f_new_nv[i] - f_target_by_idx[i])

    return {
        "Iy_A": float(dI[0]),
        "Iz_A": float(dI[1]),
        "dI_A": dI.copy(),
        "B_before_G": B0,
        "B_after_G": B_new,
        "dB_G": dB,
        "f_before_nv_GHz": f0_nv,
        "f_after_nv_GHz": f_new_nv,
        "residuals_MHz": res_MHz,
        "used_rows": sel,
        "A": A, "b_G": b, "weights": w,
    }

def pretty_print_current_plan(plan, title="Current plan to hit targets"):
    print(f"\n=== {title} ===")
    print(f"Iy (A): {plan['Iy_A']:+.6f}   Iz (A): {plan['Iz_A']:+.6f}")
    print("ΔB (G):", np.round(plan["dB_G"], 6))
    print("B_before (G):", np.round(plan["B_before_G"], 6))
    print("B_after  (G):", np.round(plan["B_after_G"], 6))
    print("\n   NV axis         f_- before (GHz)   f_- after (GHz)")
    for i, lab in enumerate(NV_LABELS):
        print(f" {lab:<12}     {plan['f_before_nv_GHz'][i]:.9f}      {plan['f_after_nv_GHz'][i]:.9f}")
    if plan["residuals_MHz"]:
        print("\nConstraint residuals (MHz) on targeted NVs:")
        for i in plan["used_rows"]:
            print(f"  NV {NV_LABELS[i]}: {plan['residuals_MHz'][i]:+,.3f} MHz")
    print()


def plan_increase_Bmag_and_predict_peaks(
    f_current_GHz,                # 4 f_- lines (any order) at current state
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    delta_Bmag_G=None,            # desired increase in |B| (Gauss), e.g. 5.0
    scale_Bmag=None,              # OR scale factor for |B|, e.g. 1.10 (10% up)
    K_crys_3x2=None,              # optional 3x2 coil map in crystal axes (G/A)
    coil_limit_A=None             # optional clamp for |Iy|,|Iz|
):
    """
    Increase the field magnitude while keeping its direction fixed,
    predict new quartet, and optionally estimate Iy/Iz to realize it.

    Returns dict with:
      - B0_G, B1_G, dB_G, |B0|, |B1|
      - f_before_nv_GHz (nv_axes order), f_after_nv_GHz
      - Iy_A, Iz_A (if K provided), and residual_dB_G (K*I - dB)
    """
    # 1) Solve current B (crystal)
    out0 = solve_B_from_odmr_order_invariant(f_current_GHz, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    B0 = out0["B"]
    B0_mag = float(np.linalg.norm(B0))
    if B0_mag <= 0:
        raise ValueError("Solved |B| is zero or invalid.")

    # 2) Decide target |B|
    if (delta_Bmag_G is None) and (scale_Bmag is None):
        raise ValueError("Provide delta_Bmag_G or scale_Bmag.")
    if (delta_Bmag_G is not None) and (scale_Bmag is not None):
        raise ValueError("Provide only one of delta_Bmag_G or scale_Bmag.")

    if delta_Bmag_G is not None:
        B1_mag = B0_mag + float(delta_Bmag_G)
    else:
        B1_mag = B0_mag * float(scale_Bmag)
    if B1_mag <= 0:
        raise ValueError("Requested |B1| <= 0; increase magnitude or use a positive scale.")

    # 3) Keep direction fixed
    B_hat = B0 / B0_mag
    B1 = B_hat * B1_mag
    dB = B1 - B0

    # 4) Predict new quartet (nv_axes order)
    f0_nv = f_minus_from_B(B0, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    f1_nv = f_minus_from_B(B1, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    plan = {
        "B0_G": B0, "B1_G": B1, "dB_G": dB,
        "B0_mag_G": B0_mag, "B1_mag_G": B1_mag,
        "f_before_nv_GHz": f0_nv,
        "f_after_nv_GHz": f1_nv,
    }

    # 5) If a coil map is provided, compute currents to realize dB:
    if K_crys_3x2 is not None:
        K = np.asarray(K_crys_3x2, float).reshape(3,2)   # 3x2
        # Least-squares (minimum-norm) solution: minimize ||K dI - dB||_2
        # Use pseudo-inverse on the left (2x3): dI = (K^T K)^{-1} K^T dB
        lhs = K.T @ K
        rhs = K.T @ dB
        dI = np.linalg.lstsq(lhs, rhs, rcond=None)[0]    # shape (2,)
        # Optional clamp
        if coil_limit_A is not None:
            dI = np.clip(dI, -abs(coil_limit_A), +abs(coil_limit_A))
        # Predicted achieved ΔB and residual
        dB_ach = K @ dI
        res = dB_ach - dB

        plan.update({
            "Iy_A": float(dI[0]),
            "Iz_A": float(dI[1]),
            "dB_achieved_G": dB_ach,
            "residual_dB_G": res,
        })

    return plan


def print_Bmag_increase_plan(plan, title="Increase |B| and predict peaks"):
    NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]
    print(f"\n=== {title} ===")
    print(f"|B|: {plan['B0_mag_G']:.6f} G  →  {plan['B1_mag_G']:.6f} G  (Δ={plan['B1_mag_G']-plan['B0_mag_G']:+.6f} G)")
    print("B0 (G):", np.round(plan["B0_G"], 6))
    print("B1 (G):", np.round(plan["B1_G"], 6))
    print("ΔB (G):", np.round(plan["dB_G"], 6))
    print("\nNV (nv_axes order)      f_- before (GHz)   f_- after (GHz)    Δf (MHz)")
    for i, lab in enumerate(NV_LABELS):
        df_MHz = 1000.0*(plan["f_after_nv_GHz"][i] - plan["f_before_nv_GHz"][i])
        print(f" {lab:<12}     {plan['f_before_nv_GHz'][i]:.9f}      {plan['f_after_nv_GHz'][i]:.9f}    {df_MHz:+8.3f}")

    if "Iy_A" in plan:
        print("\n--- Coil suggestion (LS) ---")
        print(f" Iy (A): {plan['Iy_A']:+.6f}   Iz (A): {plan['Iz_A']:+.6f}")
        print(" Achieved ΔB (G):", np.round(plan["dB_achieved_G"], 6))
        print(" Residual  ΔB (G):", np.round(plan["residual_dB_G"], 6))
        
        
        
# ---------- Predict B and peaks from currents (crystal frame) ----------
def predict_B_and_peaks_from_currents(
    Iy_A, Iz_A,
    K_crys_3x2,
    B0_G=None,                      # optional background/permanent field [Bx,By,Bz] in G
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    E_MHz=0.0                       # set >0 if you want to include strain splitting (simple approx)
):
    """
    Input:
      Iy_A, Iz_A      : coil currents (A)
      K_crys_3x2      : 3x2 coil map in crystal axes (G/A), columns = [dB/dIy, dB/dIz]
      B0_G            : optional baseline field (G). If None, uses [0,0,0]
      D_GHz, gamma_e  : spectroscopic constants
      E_MHz           : transverse strain parameter; if 0.0, returns pure Zeeman quartet

    Returns dict:
      B_coil_G        : K @ [Iy; Iz]
      B_tot_G         : B0 + B_coil
      |B|, B_hat
      projections_G   : n_i · B_tot (signed) in NV order
      f_minus_GHz     : 4 lower-branch lines (GHz) in NV order
    """
    K = np.asarray(K_crys_3x2, float).reshape(3,2)
    I = np.array([Iy_A, Iz_A], float).reshape(2)
    B0 = np.zeros(3) if B0_G is None else np.asarray(B0_G, float).reshape(3)

    # Field from coils + baseline
    B_coil = K @ I                   # (3,)
    B_tot  = B0 + B_coil
    B_mag  = float(np.linalg.norm(B_tot))
    B_hat  = B_tot / B_mag if B_mag > 0 else np.zeros(3)

    # NV projections & peaks
    n = nv_axes()
    proj = n @ B_tot                 # signed n·B (G)

    # Option A (default): no strain, just |n·B|
    if E_MHz == 0.0:
        f_minus = D_GHz - (gamma_e_MHz_per_G * np.abs(proj)) / 1000.0
    else:
        # Simple strain model: f_- = D - 0.5*sqrt((2γ|n·B|)^2 + (2E)^2)
        # (GHz units: convert MHz to GHz inside)
        gamma_GHz_per_G = gamma_e_MHz_per_G / 1000.0
        term = (2*gamma_GHz_per_G*np.abs(proj))**2 + (2*E_MHz/1000.0)**2
        f_minus = D_GHz - 0.5*np.sqrt(term)

    return {
        "B_coil_G": B_coil,
        "B_tot_G": B_tot,
        "B_mag_G": B_mag,
        "B_hat": B_hat,
        "projections_G": proj,
        "f_minus_GHz": f_minus,
    }

def print_currents_prediction(result, Iy_A, Iz_A, title="Peaks & field from currents"):
    print(f"\n=== {title} ===")
    print(f"Currents: Iy={Iy_A:+.6f} A, Iz={Iz_A:+.6f} A")
    print("B_coil (G):", np.round(result["B_coil_G"], 6))
    print("B_tot  (G):", np.round(result["B_tot_G"], 6))
    print(f"|B_tot| (G): {result['B_mag_G']:.6f}")
    print("B_hat:", np.round(result["B_hat"], 6))
    print("\n   NV axis       n·B (G, signed)     f_- (GHz)")
    for i, lab in enumerate(NV_LABELS):
        print(f" {lab:<12}  {result['projections_G'][i]:+10.6f}      {result['f_minus_GHz'][i]:.9f}")

def estimate_K_from_toggles(
    f_y_plus, f_y_minus,   # 4 f_- (any order) for Iy=+I, Iz=0 and Iy=-I, Iz=0
    f_z_plus, f_z_minus,   # 4 f_- (any order) for Iy=0, Iz=+I and Iy=0, Iz=-I
    I_A=1.0,               # amplitude used for the +/- toggles (A)
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
):
    """
    Returns K_crys (3x2) in G/A using your order-invariant solver.
    Column 0 = dB/dIy, column 1 = dB/dIz.
    """
    out_y_plus  = solve_B_from_odmr_order_invariant(f_y_plus,  D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    out_y_minus = solve_B_from_odmr_order_invariant(f_y_minus, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    out_z_plus  = solve_B_from_odmr_order_invariant(f_z_plus,  D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    out_z_minus = solve_B_from_odmr_order_invariant(f_z_minus, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    dB_y = (out_y_plus["B"] - out_y_minus["B"]) / (2.0*I_A)   # G/A
    dB_z = (out_z_plus["B"] - out_z_minus["B"]) / (2.0*I_A)   # G/A

    K = np.column_stack([dB_y, dB_z])  # shape (3,2)
    return K

def quick_explain_discrepancy(Iy_A, Iz_A, K_crys_3x2, B0_G, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    """
    Prints the predicted quartet and the per-axis projections so you can see
    exactly which NV projection grew/shrank vs your measurement.
    """
    res = predict_B_and_peaks_from_currents(Iy_A, Iz_A, K_crys_3x2, B0_G=B0_G, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    print_currents_prediction(res, Iy_A, Iz_A, title="Model vs measurement (use measured K!)")
    # Also print just the |n·B| in Gauss for easy compare with f_- via (D-f)*1000/gamma
    n = nv_axes()
    print("\n|n·B| (G) by NV axis (nv_axes order):")
    for i, lab in enumerate(NV_LABELS):
        print(f" {lab:<12}  {abs(res['projections_G'][i]):.6f} G  → f_-={res['f_minus_GHz'][i]:.6f} GHz")

# ----------------- Example -----------------
if __name__ == "__main__":
    # # try any ordering you like:
    f = [2.7666, 2.7851, 2.8222, 2.8406]  # shuffled example

    D = 2.8785
    gamma = 2.8025

    # # Rough per-amp gains from your single shot (very approximate):
    K_diag = np.array([[ 0.0,   0.0  ],
                    [-6.37,  0.0  ],   # dB/dIy ≈ (0, -6.37, 0) G/A
                    [ 0.0, -17.39 ]])  # dB/dIz ≈ (0, 0, -17.39) G/A

    # # Suppose you want to move only NV [1,1,-1] from its current to 2.8200 GHz:
    # f_base = [2.7660, 2.7851, 2.8235, 2.8405]  # any order is fine
    # target = { 1: 2.7600 }  # ori_idx 3 = [1,1,-1]

    # Iy, Iz = 1.54, 0.73
    # scale = Iy**2 + Iz**2
    # dB = np.array([ 6.1031, -9.8113, -12.6852 ])
    # dB_dIy = dB * (Iy/scale)
    # dB_dIz = dB * (Iz/scale)
    # K_min_norm = np.column_stack([dB_dIy, dB_dIz])

    # # Example: move two axes at once (weighted)
    # f_base = [2.7660, 2.7851, 2.8235, 2.8405]
    # targets = { 1: 2.7600,   # [1,1,1]
    #             3: 2.8000 }  # [1,1,-1]
    # weights = { 0: 1.0, 3: 2.0 }  # prioritize the [1,1,-1] move

    # plan2 = currents_to_hit_target_f(f_base, targets, K_min_norm, D_GHz=D, gamma_e_MHz_per_G=gamma, weights_by_idx=weights, coil_limit_A=2.0)
    # pretty_print_current_plan(plan2, "Hit two NVs with min-norm K (weights)")
    # plan = currents_to_hit_target_f(f_base, target, K_diag, D_GHz=D, gamma_e_MHz_per_G=gamma)
    # pretty_print_current_plan(plan, "Hit NV [1,1,-1] → 2.8200 GHz (diagonal K)")


    # D = 2.8785
    # gamma = 2.8025
    # f_base = [2.7660, 2.7851, 2.8235, 2.8405]  # any order

    # # Example A: increase |B| by +5 G (keep direction), just predict
    # plan = plan_increase_Bmag_and_predict_peaks(
    #     f_current_GHz=f_base,
    #     D_GHz=D, gamma_e_MHz_per_G=gamma,
    #     delta_Bmag_G=10.0,
    # )
    # print_Bmag_increase_plan(plan, "Increase |B| by +5 G")

    # # Example B: scale |B| by 1.10 (10% up) and ALSO estimate Iy/Iz with your coil map
    # # (Use your best K_crys once you fit it; below is a rough diagonal placeholder)
    # K_diag = np.array([[ 0.0,   0.0  ],
    #                 [-6.37,  0.0  ],
    #                 [ 0.0, -17.39 ]])  # G/A
    # plan2 = plan_increase_Bmag_and_predict_peaks(
    #     f_current_GHz=f_base,
    #     D_GHz=D, gamma_e_MHz_per_G=gamma,
    #     scale_Bmag=1.2,
    #     K_crys_3x2=K_diag,
    #     coil_limit_A=4.0,  # optional safety
    # )
    # print_Bmag_increase_plan(plan2, "Scale |B| by 1.10 with coil suggestion")

    # Your provisional K (crystal frame, mostly diagonal y/z)
    K_crys = np.array([
        [ 0.0,   0.0],
        [+125,  0.0],
        [ 0.0, -17.39],
    ])  # G/A

    # (Optional) your latest baseline from 4-line solve with D=2.8785
    B0 = np.array([-46.18287122, -17.44411563, -5.57779074])  # G

    # Try some currents:
    Iy, Iz = 0, 1.0

    res = predict_B_and_peaks_from_currents(Iy, Iz, K_crys, B0_G=B0, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025, E_MHz=0.0)
    print_currents_prediction(res, Iy, Iz, title="With baseline B0")

    # Coil-only case (remove permanent field):
    # res2 = predict_B_and_peaks_from_currents(1.0, 4.0, K_crys, B0_G=None)
    # print_currents_prediction(res2, 4.0, 4.0, title="Coil-only, Iy=Iz=4 A")
    
    D_GHz = 2.8785
    gamma_e = 2.8025  # MHz/G

    # ------------------ 1) Baseline: solve B0 from your 4-line spectrum ------------------
    # Fill this with your latest measured baseline quartet (any order is OK)
    f_baseline = np.array([2.7660, 2.7851, 2.8235, 2.8405], float)

    out0 = solve_B_from_odmr_order_invariant(f_baseline, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e)
    B0 = out0["B"]
    print("=== Baseline B0 (crystal) from 4-line solve ===")
    print("B0 (G):", np.round(B0, 6), "|B0| (G):", float(np.linalg.norm(B0)))
    print("f_- (nv_axes order):", np.round(out0["f_minus_nvaxes_GHz"], 9))

    # ------------------ 2) Learn K_crys from ±Iy and ±Iz toggles ------------------
    # Measure four spectra with currents:
    #  - Iy = +I, Iz = 0
    #  - Iy = -I, Iz = 0
    #  - Iy = 0, Iz = +I
    #  - Iy = 0, Iz = -I
    # Each spectrum must contain the four f_- (any order). Put your measured values below.

    # I_toggle = 1.0  # amps used for toggles

    # # >>>> TODO: replace these placeholders with your measured quartets (GHz), any order <<<<
    # f_y_plus  = np.array([2.XXX, 2.XXX, 2.XXX, 2.XXX], float)  # Iy=+I, Iz=0
    # f_y_minus = np.array([2.XXX, 2.XXX, 2.XXX, 2.XXX], float)  # Iy=-I, Iz=0
    # f_z_plus  = np.array([2.XXX, 2.XXX, 2.XXX, 2.XXX], float)  # Iy=0,  Iz=+I
    # f_z_minus = np.array([2.XXX, 2.XXX, 2.XXX, 2.XXX], float)  # Iy=0,  Iz=-I

    # K_crys = estimate_K_from_toggles(
    #     f_y_plus, f_y_minus,
    #     f_z_plus, f_z_minus,
    #     I_A=I_toggle, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e
    # )
    # print("\n=== Learned coil map K_crys (G/A) in crystal axes ===")
    # print("Columns are dB/dIy and dB/dIz:")
    # print(np.round(K_crys, 5))

    # # ------------------ 3) Predict B and peaks for any currents you enter ------------------
    # def run_prediction(Iy_A, Iz_A, title="Prediction"):
    #     res = predict_B_and_peaks_from_currents(
    #         Iy_A, Iz_A, K_crys_3x2=K_crys, B0_G=B0,
    #         D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e, E_MHz=0.0
    #     )
    #     print_currents_prediction(res, Iy_A, Iz_A, title=title)

    # # Example: reproduce your test (Iy=0, Iz=1.0)
    # run_prediction(0.0, 1.0, title="Iy=0.0 A, Iz=+1.0 A")

    # Example: try any other currents you want:
    # run_prediction(0.73, 1.54, title="Iy=0.73 A, Iz=1.54 A")