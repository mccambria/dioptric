# -*- coding: utf-8 -*-
"""
Extract the magnetic field (crystallographic axes only)
Created on March 23th, 2025
@author: Saroj Chand
"""

import numpy as np
from itertools import product, permutations

# ----------------- NV geometry -----------------
def nv_axes():
    """Four NV <111> unit vectors in the cubic {x,y,z} basis."""
    axes = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
    ], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes
NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]

# ----------------- Order-invariant solver -----------------
def solve_B_from_odmr_order_invariant(
    f_ms_minus_GHz,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    equal_tol=1e-9,
    round_decimals=12,
    sign_eps=1e-12
):
    """
    Order-invariant NV B-field solver in crystal coordinates.
    Canonicalizes ties by lexicographically smallest rounded B.
    """
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)

    # Guard: m_s=-1 line cannot exceed D (allow tiny FP slack)
    if np.any(f > D_GHz + 1e-9):
        f = np.minimum(f, D_GHz)

    n = nv_axes()  # 4x3

    # |n·B| (G) from measured f_-: |n·B| = (D - f)*1000/gamma
    abs_nB_meas = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    best_res = np.inf
    candidates = []
    for perm in permutations(range(4)):
        b_mag = abs_nB_meas[list(perm)]  # impose candidate mapping to nv_axes order
        for s in product([-1, +1], repeat=4):
            s = np.asarray(s, float)
            rhs = s * b_mag
            B, *_ = np.linalg.lstsq(n, rhs, rcond=None)
            proj = n @ B
            # robust sign-consistency
            if not np.all(proj * rhs >= -sign_eps):
                continue
            rvec = proj - rhs
            res = float(np.linalg.norm(rvec))
            candidates.append((res, B, proj, s, b_mag, perm))
            if res < best_res:
                best_res = res

    if not candidates:
        raise RuntimeError("No consistent solution found; check inputs.")

    near = [c for c in candidates if c[0] <= best_res + equal_tol]

    # Canonical, order-independent tie-break:
    def canon_key(c):
        _, B, *_ = c
        return tuple(np.round(B, round_decimals).tolist())

    res, B, proj, s, b_mag, perm = sorted(near, key=canon_key)[0]

    B_mag = float(np.linalg.norm(B))
    B_hat = B / B_mag if B_mag > 0 else np.zeros_like(B)

    # Also provide f_- and |n·B| (predicted) in nv_axes() order
    abs_nB_pred = np.abs(n @ B)
    f_nvaxes = D_GHz - (gamma_e_MHz_per_G * abs_nB_pred) / 1000.0

    return {
        "B": B,
        "B_mag": B_mag,
        "B_hat": B_hat,
        "projections": proj,            # n·B (signed), nv_axes order
        "signs": s,
        "b_magnitudes": b_mag,          # |n·B| from data, nv_axes order (per chosen perm)
        "abs_nB_pred": abs_nB_pred,     # |n·B| from B, nv_axes order (prediction)
        "f_minus_nvaxes_GHz": f_nvaxes, # predicted f_- in nv_axes order
        "residual_norm": res,
        "sign_consistent": True,
        "perm": perm                    # input idx -> nv_axes idx
    }

# ----------------- Frequency → orientation mapper -----------------
def map_frequencies_to_orientations(
    f_ms_minus_GHz, B_crystal,
    D_GHz=2.8785, gamma_e_MHz_per_G=2.8025
):
    """
    Given any-order f_- and solved B (crystal), map each measured line
    to an NV orientation minimizing |n·B| mismatch.
    """
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)
    abs_nB_meas = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    n = nv_axes()
    abs_nB_pred_all = np.abs(n @ np.asarray(B_crystal, float).reshape(3))

    # Brute-force minimal assignment
    best_perm, best_err = None, np.inf
    for perm in permutations(range(4)):
        err = 0.0
        for j in range(4):
            i = perm[j]
            d = abs_nB_meas[j] - abs_nB_pred_all[i]
            err += d*d
        if err < best_err:
            best_err, best_perm = err, perm

    rms_G = float(np.sqrt(best_err / 4.0))
    mapping = []
    for j in range(4):
        i = best_perm[j]
        mapping.append({
            "idx_meas": j,
            "f_GHz": f[j],
            "ori_idx": i,
            "ori_label": NV_LABELS[i],
            "abs_nB_meas_G": abs_nB_meas[j],
            "abs_nB_pred_G": abs_nB_pred_all[i],
        })
    return mapping, best_perm, rms_G

# ----------------- Pretty printer -----------------
def print_full_summary(f_any_order, out, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    f_any_order = np.asarray(f_any_order, float).reshape(4)
    n = nv_axes()

    print("\n========== NV ODMR → B solve (crystal frame) ==========")
    print("Input f_- (GHz), given order:", f_any_order)
    print("Chosen permutation (input idx → nv_axes idx):", out["perm"])
    print("\n--- Magnetic field (crystal axes) ---")
    print(f"B (G): {out['B']}")
    print(f"|B| (G): {out['B_mag']:.12f}")
    print(f"B_hat: {out['B_hat']}")

    print("\n--- Projections onto NV axes (nv_axes order) ---")
    for i in range(4):
        print(f" NV {NV_LABELS[i]}:  n·B = {out['projections'][i]: .6f} G   "
              f"|n·B|_meas = {out['b_magnitudes'][i]: .6f} G   "
              f"|n·B|_pred = {out['abs_nB_pred'][i]: .6f} G   "
              f"sign = {int(out['signs'][i]):+d}")

    print("\n--- Frequencies in nv_axes order (GHz) ---")
    # Using the permutation, reorder the *measured* lines into nv_axes order:
    f_nvorder = np.empty(4, float)
    for in_idx, nv_idx in enumerate(out["perm"]):
        f_nvorder[nv_idx] = f_any_order[in_idx]
    print(" measured f_- (nv_axes order):", np.round(f_nvorder, 12))
    print(" predicted f_- (nv_axes order):", np.round(out["f_minus_nvaxes_GHz"], 12))

    print("\n--- Fit quality ---")
    print(f"Residual norm: {out['residual_norm']:.3e}   |   sign_consistent: {out['sign_consistent']}")

    # Also show a per-line mapping table in the original input order
    mapping, perm_chk, rmsG = map_frequencies_to_orientations(f_any_order, out["B"], D_GHz, gamma_e_MHz_per_G)
    perm_chk == out["perm"]
    print("\n--- Frequency → Orientation mapping (original input order) ---")
    for m in mapping:
        print(f" meas idx {m['idx_meas']}: f_-={m['f_GHz']:.9f} GHz  →  NV {m['ori_label']} "
              f"(ori idx {m['ori_idx']});  |n·B|_meas={m['abs_nB_meas_G']:.6f} G, "
              f"|n·B|_pred={m['abs_nB_pred_G']:.6f} G")
    print(f" RMS(|n·B| mismatch) ≈ {rmsG:.6e} G")
    print("========================================================\n")

# ----------------- Example -----------------
if __name__ == "__main__":
    # try any ordering you like:
    # f = [2.7666, 2.7851, 2.8222, 2.8406]  # shuffled example
    f = [2.7230, 2.7461, 2.8300, 2.8475]  # shuffled example

    out = solve_B_from_odmr_order_invariant(f)
    print_full_summary(f, out)


    # Optional: compare to your baseline B0 from the 4-line solve to get ΔB
    # (Assuming you computed out0 from the full baseline quartet earlier)
    # dB = out3["B"] - out0["B"]
    # print("ΔB (G) from this setting vs baseline:", np.round(dB, 6))
    # print("Δf (MHz) in nv_axes order:",
    #       np.round(1000*(out0["f_minus_nvaxes_GHz"] - out3["f_minus_nvaxes_GHz"]), 6))

