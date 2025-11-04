import numpy as np

# --- reuse your nv_axes() if already defined ---
def nv_axes():
    axes = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
    ], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes

def f_minus_from_B(B_vec_G, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
    """
    Return the 4 lower-branch ODMR lines (ms=0->-1) in GHz.
    Convention: f_- = D - gamma_e * |n·B| / 1000  (always <= D).
    """
    n = nv_axes()             # (4,3)
    projs_abs_G = np.abs(n @ np.asarray(B_vec_G, float).reshape(3))
    return D_GHz - (gamma_e_MHz_per_G * projs_abs_G) / 1000.0

# ----- coil models -----
def apply_currents_diag(B_bg_G, Ixy_A, kx_G_per_A=20.0, ky_G_per_A=20.0):
    Ix, Iy = np.asarray(Ixy_A, float)
    B_coil = np.array([kx_G_per_A * Ix, ky_G_per_A * Iy, 0.0])
    return np.asarray(B_bg_G, float).reshape(3) + B_coil

def apply_currents_general(B_bg_G, Ixy_A, K_3x2):
    return np.asarray(B_bg_G, float).reshape(3) + np.asarray(K_3x2, float) @ np.asarray(Ixy_A, float).reshape(2)

def currents_zero_transverse(B_bg_G, kx_G_per_A=20.0, ky_G_per_A=20.0):
    Bx, By, _ = np.asarray(B_bg_G, float).reshape(3)
    Ix = -Bx / kx_G_per_A
    Iy = -By / ky_G_per_A
    return np.array([Ix, Iy])

def currents_set_inplane(B_bg_G, Bperp_G, theta_deg, kx_G_per_A=20.0, ky_G_per_A=20.0):
    """
    Target in-plane field of magnitude Bperp at angle theta (deg) from +x toward +y.
    Keep Bz equal to background (no z coil).
    """
    th = np.deg2rad(theta_deg)
    B_tgt = np.array([Bperp_G*np.cos(th), Bperp_G*np.sin(th), np.asarray(B_bg_G, float).reshape(3)[2]])
    Ix = (B_tgt[0] - B_bg_G[0]) / kx_G_per_A
    Iy = (B_tgt[1] - B_bg_G[1]) / ky_G_per_A
    return np.array([Ix, Iy]), B_tgt

# ----- end-to-end helper -----
def predict_f_minus_after_coils(B_bg_G,
                                mode="zero_transverse",
                                kx_G_per_A=20.0,
                                ky_G_per_A=20.0,
                                Bperp_G=None,
                                theta_deg=None,
                                Ixy_override=None,
                                K_3x2=None,
                                D_GHz=2.870,
                                gamma_e_MHz_per_G=2.8025):
    """
    Compute the new f_- list after applying coil currents.

    Parameters
    ----------
    B_bg_G : array-like, shape (3,)
        Background field (G) from your solver.
    mode : {"zero_transverse","set_inplane","manual"}
        - "zero_transverse": cancels Bx, By.
        - "set_inplane": set in-plane magnitude/angle (needs Bperp_G, theta_deg).
        - "manual": directly use Ixy_override currents (A).
    kx_G_per_A, ky_G_per_A : float
        Coil gains for x and y (G/A) if using diagonal model.
    Bperp_G, theta_deg : float
        Target in-plane magnitude (G) and angle (deg) for "set_inplane".
    Ixy_override : (Ix, Iy) in amps for "manual".
    K_3x2 : optional 3x2 matrix (G/A). If provided, uses general model (misalignment).
    D_GHz, gamma_e_MHz_per_G : floats
        Spectroscopic constants.

    Returns
    -------
    out : dict with keys
        "Ixy_A" (amps), "B_tot_G" (3,), "f_minus_GHz" (4,), "model" ("diag" or "general")
    """
    B_bg_G = np.asarray(B_bg_G, float).reshape(3)

    # choose currents
    if mode == "zero_transverse":
        Ixy_A = currents_zero_transverse(B_bg_G, kx_G_per_A, ky_G_per_A)
    elif mode == "set_inplane":
        assert Bperp_G is not None and theta_deg is not None, "Provide Bperp_G and theta_deg"
        Ixy_A, _ = currents_set_inplane(B_bg_G, Bperp_G, theta_deg, kx_G_per_A, ky_G_per_A)
    elif mode == "manual":
        assert Ixy_override is not None, "Provide Ixy_override=(Ix,Iy) in amps"
        Ixy_A = np.asarray(Ixy_override, float).reshape(2)
    else:
        raise ValueError("mode must be one of {'zero_transverse','set_inplane','manual'}")

    # propagate to total B
    if K_3x2 is None:
        model = "diag"
        B_tot_G = apply_currents_diag(B_bg_G, Ixy_A, kx_G_per_A, ky_G_per_A)
    else:
        model = "general"
        B_tot_G = apply_currents_general(B_bg_G, Ixy_A, K_3x2)

    # predict lower-branch lines
    f_minus = f_minus_from_B(B_tot_G, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    return {"Ixy_A": Ixy_A, "B_tot_G": B_tot_G, "f_minus_GHz": f_minus, "model": model}

# ================== examples ==================
if __name__ == "__main__":
    # Background field from your earlier solve:
    B_bg = np.array([-6.18037755, -18.54113264, -43.26264283])  # G

    # A) Zero Bx, By
    outA = predict_f_minus_after_coils(B_bg, mode="zero_transverse", kx_G_per_A=15.0, ky_G_per_A=15.0)
    print("[Zero transverse] Ix,Iy (A):", outA["Ixy_A"])
    print(" New B_tot (G):", outA["B_tot_G"])
    print(" f_minus (GHz):", outA["f_minus_GHz"])

    # B) Set B_perp = 10 G along +x (theta=0°), keep Bz fixed
    outB = predict_f_minus_after_coils(B_bg, mode="set_inplane",
                                       Bperp_G=10.0, theta_deg=0.0,
                                       kx_G_per_A=15.0, ky_G_per_A=15.0)
    print("\n[Set B_perp=10 G at 0°] Ix,Iy (A):", outB["Ixy_A"])
    print(" New B_tot (G):", outB["B_tot_G"])
    print(" f_minus (GHz):", outB["f_minus_GHz"])

    # C) Manual currents (e.g., try +0.2 A on both)
    outC = predict_f_minus_after_coils(B_bg, mode="manual", Ixy_override=(2, 2))
    print("\n[Manual currents 0.2A,0.2A] Ix,Iy (A):", outC["Ixy_A"])
    print(" New B_tot (G):", outC["B_tot_G"])
    print(" f_minus (GHz):", outC["f_minus_GHz"])

    # D) If you have a measured K (uncomment and replace numbers), it will use the general model
    K = np.array([[19.8,  0.3],
                  [ 0.5, 20.2],
                  [ 0.1, -0.2]])  # G/A, example
    outD = predict_f_minus_after_coils(B_bg, mode="set_inplane",
                                       Bperp_G=15, theta_deg=55.0,
                                       K_3x2=K)  # general model
    print("\n[General K, 20 G @45°] Ix,Iy (A):", outD['Ixy_A'])
    print(" New B_tot (G):", outD['B_tot_G'])
    print(" f_minus (GHz):", outD['f_minus_GHz'])


# import numpy as np

# # --- NV axes (same order as your solver) ---
# def nv_axes():
#     axes = np.array([
#         [ 1,  1,  1],
#         [-1,  1,  1],
#         [ 1, -1,  1],
#         [ 1,  1, -1],
#     ], dtype=float)
#     axes /= np.linalg.norm(axes, axis=1, keepdims=True)
#     return axes

# def f_minus_from_B(B_vec_G, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
#     """Return 4 lower-branch ODMR lines (GHz): f_- = D - gamma_e*|n·B|/1000."""
#     n = nv_axes()  # (4,3)
#     projs_abs_G = np.abs(n @ np.asarray(B_vec_G, float).reshape(3))
#     return D_GHz - (gamma_e_MHz_per_G * projs_abs_G) / 1000.0

# # --- Coil models (diagonal gains; no z-coil) ---
# def apply_currents_diag(B_bg_G, Ixy_A, kx_G_per_A=20.0, ky_G_per_A=20.0):
#     Ix, Iy = np.asarray(Ixy_A, float)
#     B_coil = np.array([kx_G_per_A * Ix, ky_G_per_A * Iy, 0.0])
#     return np.asarray(B_bg_G, float).reshape(3) + B_coil

# # === Core: increase splitting of a chosen axis j by a desired amount ===
# def increase_axis_splitting(
#     B_bg_G,
#     j_axis,
#     delta_split_MHz=None,
#     delta_abs_proj_G=None,
#     Ixy_base_A=(0.0, 0.0),
#     kx_G_per_A=20.0,
#     ky_G_per_A=20.0,
#     D_GHz=2.870,
#     gamma_e_MHz_per_G=2.8025,
# ):
#     """
#     Increase the splitting (lower f_-) of NV line j by nudging Ix,Iy along a direction
#     that maximizes |n_j · ΔB| and minimally perturbs others (in 2D coil space).

#     Choose either:
#       - delta_split_MHz (desired *decrease* in f_- for axis j, in MHz), or
#       - delta_abs_proj_G (desired *increase* in |n_j·B|, in Gauss).

#     Returns dict with initial & new currents, fields, and f_- lists.
#     """
#     assert (delta_split_MHz is not None) ^ (delta_abs_proj_G is not None), \
#         "Specify exactly one of delta_split_MHz or delta_abs_proj_G."

#     B_bg_G = np.asarray(B_bg_G, float).reshape(3)
#     I0 = np.asarray(Ixy_base_A, float).reshape(2)
#     n = nv_axes()
#     kx, ky = float(kx_G_per_A), float(ky_G_per_A)

#     # Current total B and initial f_-
#     B0 = apply_currents_diag(B_bg_G, I0, kx, ky)
#     f0 = f_minus_from_B(B0, D_GHz, gamma_e_MHz_per_G)

#     # We want to *increase* |n_j · B| so that f_- decreases by delta_split_MHz (if given)
#     pj = float(n[j_axis] @ B0)                 # current projection for axis j
#     sgn = 1.0 if pj >= 0 else -1.0            # push farther from zero: Δp_j = sgn * Δ|p|
#     if delta_abs_proj_G is None:
#         # convert MHz split change to Gauss change in |n·B|
#         delta_abs_proj_G = float(delta_split_MHz) / gamma_e_MHz_per_G * 1000.0
#     delta_pj = sgn * delta_abs_proj_G          # desired change in the signed projection

#     # Choose ΔI along vector that maximizes Δp_j per amp in 2D: v ∝ (n_jx/kx, n_jy/ky)
#     # This yields Δp_j = (n_jx^2 + n_jy^2) * α  with  ΔI = α * (n_jx/kx, n_jy/ky)
#     njx, njy = n[j_axis, 0], n[j_axis, 1]
#     denom = (njx**2 + njy**2)
#     if denom < 1e-12:
#         raise ValueError("Selected NV axis has no x/y component; cannot steer with x/y coils only.")
#     alpha = delta_pj / denom
#     dIx = alpha * (njx / kx)
#     dIy = alpha * (njy / ky)
#     I_new = I0 + np.array([dIx, dIy])

#     # New field & lines
#     B_new = apply_currents_diag(B_bg_G, I_new, kx, ky)
#     f_new = f_minus_from_B(B_new, D_GHz, gamma_e_MHz_per_G)

#     return {
#         "Ixy_base_A": I0,
#         "Ixy_new_A": I_new,
#         "delta_Ixy_A": np.array([dIx, dIy]),
#         "B_base_G": B0,
#         "B_new_G": B_new,
#         "f_minus_initial_GHz": f0,    # initial positions (same order as nv_axes)
#         "f_minus_new_GHz": f_new,     # after the nudge
#         "axis_index": int(j_axis),
#         "delta_abs_proj_G": float(abs(delta_abs_proj_G)),
#         "applied_signed_proj_change_G": float(delta_pj),
#     }

# # ================== Example: increase splitting of the ~2.78 GHz line ==================
# if __name__ == "__main__":
#     # Your background field (from your solver output):
#     B_bg = np.array([-6.18037755, -18.54113264, -43.26264283])  # G

#     # Identify which index is ~2.78 GHz at zero added current (coils off)
#     f_init = f_minus_from_B(B_bg)
#     print("Initial f_- (GHz) in nv_axes() order:", f_init)

#     # Suppose the element near 2.78 GHz is at index j=1 (adjust if your order differs):
#     j = int(np.argmin(np.abs(f_init - 2.78)))  # auto-pick closest to 2.78 GHz

#     # Option A: ask for +5 MHz more splitting (i.e., push f_- down by 0.005 GHz = 5 MHz)
#     out = increase_axis_splitting(
#         B_bg_G=B_bg,
#         j_axis=j,
#         delta_split_MHz=5.0,     # desired *decrease* in f_- for that axis
#         Ixy_base_A=(0.0, 0.0),   # base currents; change if you already bias the coils
#         kx_G_per_A=20.0,
#         ky_G_per_A=20.0,
#     )

#     print(f"\nTarget axis index: {out['axis_index']}")
#     print("Base currents [Ix, Iy] (A):", out["Ixy_base_A"])
#     print("New  currents [Ix, Iy] (A):", out["Ixy_new_A"])
#     print("Δ currents [A]:", out["delta_Ixy_A"])
#     print("B_base (G):", out["B_base_G"])
#     print("B_new  (G):", out["B_new_G"])
#     print("\nInitial f_- (GHz):", out["f_minus_initial_GHz"])  # printed in same order
#     print("New     f_- (GHz):", out["f_minus_new_GHz"])
