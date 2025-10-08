# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""

import numpy as np
from itertools import product

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

def solve_B_from_odmr(f_ms_minus_GHz, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
    """
    Given four m_s=-1 ODMR lines (GHz) for the four NV orientations,
    recover the 3D magnetic field vector B (in Gauss), its magnitude, unit vector,
    and the per-axis projections and signs that best fit.

    Parameters
    ----------
    f_ms_minus_GHz : iterable of length 4
        Observed |0>→|-1> transition frequencies in GHz (one per NV axis).
        Order must correspond to nv_axes() order.
    D_GHz : float
        Zero-field splitting (GHz).
    gamma_e_MHz_per_G : float
        Electron gyromagnetic ratio (MHz/G).

    Returns
    -------
    result : dict
        Keys: B, B_mag, B_hat, projections, signs, residual, D_eff_list
    """
    f = np.asarray(f_ms_minus_GHz, dtype=float)
    assert f.shape == (4,), "Need exactly 4 frequencies (one per NV axis)."

    # Step 1–2: frequency shifts and projection magnitudes
    # b_i = (D - f_i)/gamma_e  (Gauss)
    shifts_MHz = (D_GHz - f) * 1000.0
    b_mag = shifts_MHz / gamma_e_MHz_per_G  # positive magnitudes
    n = nv_axes()                            # 4x3

    # We must determine signs s_i ∈ {+1,-1} so that  n_i · B = s_i * b_mag_i
    # With a chosen sign pattern s, solve least-squares:  N B ≈ s ∘ b_mag
    # (N is 4x3, B is 3x1, right-hand side is 4x1).  Try all 2^4 patterns.

    best = None
    for s in product([-1, +1], repeat=4):               # all sign assignments
        s = np.array(s, dtype=float)                    # shape (4,)
        rhs = s * b_mag                                 # shape (4,)

        # Least-squares solution of N B = rhs
        B, residuals, _, _ = np.linalg.lstsq(n, rhs, rcond=None)

        # Compute consistency: dot products and residual norm
        proj = n @ B                                    # predicted projections
        # residual vector (not just sum-of-squares that lstsq reports)
        rvec = proj - rhs
        res_norm = np.linalg.norm(rvec)

        # Optional: sanity—ensure signs of proj match s (within small tolerance)
        sign_ok = np.all(np.sign(proj + 1e-12) == np.sign(rhs + 1e-12))

        score = res_norm + (0.0 if sign_ok else 1e3)    # heavily penalize sign flips

        if (best is None) or (score < best["score"]):
            best = dict(
                B=B,
                projections=proj,
                signs=s,
                rhs=rhs,
                res_norm=res_norm,
                sign_ok=sign_ok,
                score=score,
            )

    # Final nice-to-haves
    B = best["B"]
    B_mag = float(np.linalg.norm(B))
    B_hat = B / B_mag if B_mag > 0 else np.zeros_like(B)

    return {
        "B": B,                                # (Gx, Gy, Gz) in Gauss
        "B_mag": B_mag,                        # |B| in Gauss
        "B_hat": B_hat,                        # unit vector
        "projections": best["projections"],    # n_i · B (Gauss)
        "signs": best["signs"],                # chosen sign pattern
        "b_magnitudes": b_mag,                 # |n_i · B| (Gauss) from data
        "residual_norm": best["res_norm"],     # ||N B - s*b||_2
        "sign_consistent": best["sign_ok"],
    }

# ---------------- Example with your numbers ----------------
f_ms_minus = [2.76, 2.78, 2.82, 2.84]  # GHz, ordered as nv_axes()
out = solve_B_from_odmr(f_ms_minus, D_GHz=2.870, gamma_e_MHz_per_G=2.8025)

print("B (G):", out["B"])
print("|B| (G):", out["B_mag"])
print("B_hat:", out["B_hat"])
print("Projections n·B (G):", out["projections"])
print("Chosen signs:", out["signs"])
print("Abs projections from data (G):", out["b_magnitudes"])
print("Residual norm:", out["residual_norm"], "| sign-consistent:", out["sign_consistent"])
