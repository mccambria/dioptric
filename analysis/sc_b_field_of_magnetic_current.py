import numpy as np
import sys
from sc_b_field_of_magnetic import solve_B_from_odmr_order_invariant, print_full_summary
# ---------------- NV geometry in CRYSTAL frame ----------------
def nv_axes():
    a = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
    ], dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)

def projections_abs_G(B_vec_G):
    n = nv_axes()
    return np.abs(n @ np.asarray(B_vec_G, float).reshape(3))

def df_GHz_from_absproj(absproj_G, gamma_e_MHz_per_G=2.8025):
    absproj_G = np.asarray(absproj_G, float).reshape(4)
    return (gamma_e_MHz_per_G * absproj_G) / 1000.0

def f_minus_from_B(B_vec_G, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
    return D_GHz - df_GHz_from_absproj(projections_abs_G(B_vec_G), gamma_e_MHz_per_G)

# ---------------- Coil models: y and z only ----------------
# ================= Coils (crystal y/z) utilities =================

NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]

def apply_currents_yz_diag(B0_G, Iy_A, Iz_A, ky_G_per_A=20.0, kz_G_per_A=20.0):
    """Ideal diagonal coil map: ΔB = (0, ky*Iy, kz*Iz)."""
    B0 = np.asarray(B0_G, float).reshape(3)
    dB = np.array([0.0, ky_G_per_A*Iy_A, kz_G_per_A*Iz_A], float)
    return B0 + dB

def apply_currents_general(B0_G, Iyz_A, K_crys_3x2):
    """General (possibly signed/crosstalk) crystal-frame map: B = B0 + K @ [Iy, Iz]."""
    B0 = np.asarray(B0_G, float).reshape(3)
    K  = np.asarray(K_crys_3x2, float).reshape(3,2)
    I  = np.asarray(Iyz_A, float).reshape(2)
    return B0 + K @ I

def currents_for_ByBz_targets_diag(B0_G, By_tgt_G, Bz_tgt_G, ky_G_per_A=20.0, kz_G_per_A=20.0):
    """Pick Iy,Iz to hit target By,Bz for diagonal gains (no x coil)."""
    B0 = np.asarray(B0_G, float).reshape(3)
    Iy = (By_tgt_G - B0[1]) / ky_G_per_A
    Iz = (Bz_tgt_G - B0[2]) / kz_G_per_A
    return np.array([Iy, Iz], float)

def currents_for_ByBz_targets_general(B0_G, By_tgt_G, Bz_tgt_G, K_crys_3x2):
    """
    Solve least-squares for Iy,Iz to achieve By,Bz with general K.
    Uses only y,z rows of K (2x2 effective).
    """
    B0 = np.asarray(B0_G, float).reshape(3)
    K  = np.asarray(K_crys_3x2, float).reshape(3,2)
    A  = K[1:3, :]                       # rows for y,z
    b  = np.array([By_tgt_G - B0[1], Bz_tgt_G - B0[2]], float)
    Iyz, *_ = np.linalg.lstsq(A, b, rcond=None)
    return Iyz

def set_Bperp_theta(B0_G, Bperp_G, theta_deg):
    """Utility: desired (By,Bz) from magnitude/angle in y–z plane (θ from +y toward +z)."""
    th = np.deg2rad(theta_deg)
    By_tgt = Bperp_G * np.cos(th)
    Bz_tgt = Bperp_G * np.sin(th)
    return By_tgt, Bz_tgt

def f_minus_from_B(B_vec_G, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
    n = nv_axes()
    projs_abs = np.abs(n @ np.asarray(B_vec_G, float).reshape(3))
    return D_GHz - (gamma_e_MHz_per_G * projs_abs) / 1000.0

# ---------- Splitting control with y/z coils ----------
def nudge_axis_split_yz(
    B0_G, axis_idx, delta_f_MHz,
    K_crys_3x2=None, ky_G_per_A=20.0, kz_G_per_A=20.0,
    D_GHz=2.870, gamma_e_MHz_per_G=2.8025
):
    """
    Increase splitting of one NV axis by delta_f_MHz (decrease f_- by that amount),
    using only y/z coils. Linearized step assuming the sign of n·B does not flip.
    Minimal-norm solution.
    """
    n = nv_axes()
    B0 = np.asarray(B0_G, float).reshape(3)
    proj = float(n[axis_idx] @ B0)
    sgn  = 1.0 if proj >= 0 else -1.0

    # target increase in |n·B| (Gauss)
    delta_abs_G = (delta_f_MHz / gamma_e_MHz_per_G) * 1000.0
    # scalar constraint: n·(ΔB) = sgn * delta_abs_G
    # With y/z coils: ΔB = K @ ΔI (general) or ΔB=(0, ky dIy, kz dIz) (diag)
    if K_crys_3x2 is None:
        # diag map
        A = np.array([n[axis_idx,1]*ky_G_per_A, n[axis_idx,2]*kz_G_per_A], float).reshape(1,2)
    else:
        A = (n[axis_idx].reshape(1,3) @ np.asarray(K_crys_3x2, float)).reshape(1,2)
    b = np.array([sgn * delta_abs_G], float)  # 1x1

    # Minimal-norm ΔI solving A ΔI = b
    dI, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Apply
    if K_crys_3x2 is None:
        B_new = apply_currents_yz_diag(B0, dI[0], dI[1], ky_G_per_A, kz_G_per_A)
    else:
        B_new = apply_currents_general(B0, dI, K_crys_3x2)

    return dict(
        delta_Iyz_A=dI,
        B_new_G=B_new,
        f_minus_new_GHz=f_minus_from_B(B_new, D_GHz, gamma_e_MHz_per_G)
    )

def target_two_axes_splits_yz(
    B0_G, axes_idx, target_f_minus_GHz,
    K_crys_3x2=None, ky_G_per_A=20.0, kz_G_per_A=20.0,
    D_GHz=2.870, gamma_e_MHz_per_G=2.8025
):
    """
    Hit two NV axes' f_- targets (least-squares) using y/z coils only.
    We linearize around current signs.
    """
    assert len(axes_idx) == 2 and len(target_f_minus_GHz) == 2
    n = nv_axes()
    B0 = np.asarray(B0_G, float).reshape(3)

    # current signs for the chosen axes
    s = np.sign(n @ B0 + 1e-15)  # shape (4,)

    # desired |n·B| for the two axes (Gauss)
    abs_des = [(D_GHz - f)*1000.0/gamma_e_MHz_per_G for f in target_f_minus_GHz]

    # Build 2x2 system in ΔI
    if K_crys_3x2 is None:
        # diag: Δp_i = n_i,y ky dIy + n_i,z kz dIz
        A = np.array([
            [n[axes_idx[0],1]*ky_G_per_A, n[axes_idx[0],2]*kz_G_per_A],
            [n[axes_idx[1],1]*ky_G_per_A, n[axes_idx[1],2]*kz_G_per_A],
        ], float)
    else:
        A = (n[axes_idx] @ np.asarray(K_crys_3x2, float)).reshape(2,2)

    p0 = (n @ B0)[axes_idx]              # signed current projections for the two axes
    b  = np.array([s[axes_idx[0]]*abs_des[0] - p0[0],
                   s[axes_idx[1]]*abs_des[1] - p0[1]], float)

    dI, *_ = np.linalg.lstsq(A, b, rcond=None)

    # apply
    if K_crys_3x2 is None:
        B_new = apply_currents_yz_diag(B0, dI[0], dI[1], ky_G_per_A, kz_G_per_A)
    else:
        B_new = apply_currents_general(B0, dI, K_crys_3x2)

    return dict(
        delta_Iyz_A=dI,
        B_new_G=B_new,
        f_minus_new_GHz=f_minus_from_B(B_new, D_GHz, gamma_e_MHz_per_G)
    )

# ================= Pretty printers =================

def print_coil_step(title, out_Bsolve, B_after, Iyz, D_GHz=2.870, gamma_e_MHz_per_G=2.8025):
    n = nv_axes()
    f0 = out_Bsolve["f_minus_nvaxes_GHz"]
    f1 = f_minus_from_B(B_after, D_GHz, gamma_e_MHz_per_G)
    print(f"\n[{title}]")
    print(" Iy,Iz (A):", np.round(np.asarray(Iyz, float), 6))
    print(" B_before (G):", np.round(out_Bsolve["B"], 6))
    print(" B_after  (G):", np.round(B_after, 6))
    print(" f_- before (nv_axes order):", np.round(f0, 12))
    print(" f_- after  (nv_axes order):", np.round(f1, 12))
    print(" Δf (MHz, nv_axes order):", np.round(1000*(f0 - f1), 6))


# ===== utilities: projections from lines (robust to D drift if both branches available) =====
def abs_proj_from_fminus(f_minus_GHz, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    """|n·B| (G) from lower branch only."""
    return (D_GHz - float(f_minus_GHz)) * 1000.0 / float(gamma_e_MHz_per_G)

def abs_proj_from_branches(f_minus_GHz, f_plus_GHz, gamma_e_MHz_per_G=2.8025):
    """|n·B| (G) from the pair (f-, f+), cancels D:  f+ - f- = 2*gamma*|n·B|/1000."""
    return (float(f_plus_GHz) - float(f_minus_GHz)) * 1000.0 / (2.0 * float(gamma_e_MHz_per_G))

# ===== K_crys fitter: B = B0 + K @ Iyz, with Iyz = [Iy, Iz]^T =====
def fit_K_crys_from_measurements(meas_list, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    """
    Fit K_crys (3x2) from multiple measurements:
      meas = dict(
        Iy=float, Iz=float,
        f_minus=[4 floats]  OR  branches=[(f_-0,f_+0), (f_-1,f_+1), (f_-2,f_+2), (f_-3,f_+3)]
      )
    Assumes all lines are from the same spot; uses your order-invariant solver to get B for each meas.
    Returns: K (3x2), B0 (reference B), residuals dict.
    """
    assert len(meas_list) >= 2, "Need ≥2 measurements (ref + another) to estimate K."
    # 1) Solve B for each measurement
    B_list, I_list = [], []
    for m in meas_list:
        if "branches" in m and m["branches"] is not None:
            # Optional sanity: we could compute |n·B| from pairs, but we still solve via f_- list.
            f_minus = [fm for (fm, fp) in m["branches"]]
        else:
            f_minus = m["f_minus"]
        out = solve_B_from_odmr_order_invariant(f_minus, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
        B_list.append(out["B"])
        I_list.append(np.array([m["Iy"], m["Iz"]], float))
    B_arr = np.vstack(B_list)      # (M,3)
    I_arr = np.vstack(I_list)      # (M,2)

    # 2) Pick the first as reference (could also average the ones with Iy=Iz=0 if you have many)
    B0 = B_arr[0]
    dB = (B_arr - B0)              # (M,3)
    # Build tall system for all rows (x,y,z) at once:
    # For each measurement k: dB_k (3,) = K (3x2) @ I_k (2,)
    # Stack as: [I_k^T ⊗ I_3] vec(K) = dB_k, solve in LS sense
    # Easier: solve row-wise: for each axis j, dB[:,j] ~ I_arr @ K_rowj
    K = np.zeros((3,2), float)
    for j in range(3):
        Kj, *_ = np.linalg.lstsq(I_arr, dB[:, j], rcond=None)
        K[j, :] = Kj

    # 3) Residuals (how well K reproduces all dB)
    dB_pred = I_arr @ K.T
    res = dB - dB_pred
    rms = float(np.sqrt(np.mean(res**2)))
    return K, B0, dict(rms_G=rms, dB=dB, dB_pred=dB_pred, res=res, I_arr=I_arr)

def print_K_summary(K):
    print("\n=== Estimated K_crys (G/A) ===")
    print("columns = effect of 1 A on [Iy, Iz]")
    print("   dB/dIy:", np.round(K[:,0], 6), " (→ ΔB for +1 A on y-coil)")
    print("   dB/dIz:", np.round(K[:,1], 6), " (→ ΔB for +1 A on z-coil)")
    # quick axis sanity
    dom_y = np.argmax(np.abs(K[:,0]))
    dom_z = np.argmax(np.abs(K[:,1]))
    ax = ["Bx","By","Bz"]
    print(f" dominant component from Iy is {ax[dom_y]}")
    print(f" dominant component from Iz is {ax[dom_z]}")

# ===== convenience: solve B before/after and show coil-induced ΔB =====
def summarize_one_setting(B_before_G, Iy, Iz, D_GHz, gamma_e_MHz_per_G, f_minus_after):
    out_after = solve_B_from_odmr_order_invariant(f_minus_after, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    B_after = out_after["B"]
    dB = B_after - np.asarray(B_before_G, float).reshape(3)
    print("\n--- Coil setting summary ---")
    print(f" Currents Iy={Iy:.4f} A, Iz={Iz:.4f} A")
    print(" B_before (G):", np.round(B_before_G, 6))
    print(" B_after  (G):", np.round(B_after, 6))
    print(" ΔB       (G):", np.round(dB, 6))
    return out_after, dB

if __name__ == "__main__":
    # ---------------- 1) Base solve from your measured quartet ----------------
    D = 2.8785
    gamma = 2.8025  # MHz/G

    f_base = [2.7660, 2.7851, 2.8235, 2.8405]  # any order
    out0 = solve_B_from_odmr_order_invariant(f_base, D_GHz=D, gamma_e_MHz_per_G=gamma)
    # print_full_summary(f_base, out0)  # nice one-shot summary
    B0 = out0["B"]

    # ---------------- 2) One coil setting: Iy=1.54 A, Iz=0.73 A ---------------
    # (You should measure ALL 4 after-lines; put them here.)
    # If you only have two, still run this; you’ll need a full quartet later to fit K.
    f_after = [
        2.7991,  # example (you reported for one orientation)
        2.8282,  # example (you reported for one orientation)
        2.8697,  # placeholder
        # 2.7950,  # placeholder
    ]
    out1, dB_1 = summarize_one_setting(B0, Iy=1.54, Iz=0.73, D_GHz=D, gamma_e_MHz_per_G=gamma, f_minus_after=f_after)

    # ---------------- 3) Collect measurements to fit K_crys -------------------
    # You need at least 2 distinct current settings (ref + another). Best is 3+.
    # Prepare a list of dicts: each has Iy, Iz, and a full quartet of f_-.
    meas_list = [
        {"Iy": 0.0,  "Iz": 0.0,  "f_minus": f_base},   # reference
        {"Iy": 1.54, "Iz": 0.73, "f_minus": f_after},  # your mixed setting
        # Add at least one more distinct setting (ideally one-coil-at-a-time):
        # e.g., Iy sweep, Iz=0
        # {"Iy": 1.00, "Iz": 0.0,  "f_minus": [ ... four after-lines ... ]},
        # {"Iy": 0.0,  "Iz": 1.00, "f_minus": [ ... four after-lines ... ]},
    ]

    # Fit K (3x2). With only 2 entries you’ll get a first estimate; 3+ is better.
    K, B0_fit, stats = fit_K_crys_from_measurements(meas_list, D_GHz=D, gamma_e_MHz_per_G=gamma)
    print_K_summary(K)
    print("Fit RMS (G):", stats["rms_G"])

    # ---------------- 4) Use K to predict / plan coil moves -------------------
    # Example: set By,Bz targets (diagonal or general map).
    # If you trust K (general), do:
    By_tgt, Bz_tgt = 10.0, -20.0
    Iyz = currents_for_ByBz_targets_general(B0, By_tgt, Bz_tgt, K_crys_3x2=K)
    B_new = apply_currents_general(B0, Iyz, K)
    print_coil_step("Hit By,Bz targets with fitted K", out0, B_new, Iyz, D_GHz=D, gamma_e_MHz_per_G=gamma)

    # Or: nudge the splitting of a specific NV axis by Δf (MHz) using y/z coils:
    axis_to_nudge = 0  # 0→[1,1,1], 1→[-1,1,1], 2→[1,-1,1], 3→[1,1,-1]
    nudge = nudge_axis_split_yz(B0, axis_to_nudge, delta_f_MHz=5.0, K_crys_3x2=K, D_GHz=D, gamma_e_MHz_per_G=gamma)
    print_coil_step(f"Nudge NV axis {axis_to_nudge} by +5 MHz (lower branch)", out0, nudge["B_new_G"], nudge["delta_Iyz_A"], D_GHz=D, gamma_e_MHz_per_G=gamma)
sys.exit()
# ================= Demo / how-to =================
if __name__ == "__main__":
    # 1) Solve B in crystal axes from any-order input
    f_input = [2.76, 2.78, 2.82, 2.84]      # try shuffling; result is invariant
    out = solve_B_from_odmr_order_invariant(f_input)
    # print_full_summary(f_input, out)

    # Example coil model: choose ONE of these
    # (A) Simple diagonal 20 G/A
    ky, kz = 20.0, 20.0
    K = None

    # (B) Or measured general K (uncomment to use)
    # K = np.array([[  0.2,  -0.3],
    #               [ 19.7,   0.5],
    #               [ -0.1,  20.3]], float)  # example G/A (columns = Iy, Iz)

    B0 = out["B"]

    # 2) Zero By,Bz
    if K is None:
        IyIz = currents_for_ByBz_targets_diag(B0, 0.0, 0.0, ky, kz)
        Bz0 = apply_currents_yz_diag(B0, IyIz[0], IyIz[1], ky, kz)
    else:
        IyIz = currents_for_ByBz_targets_general(B0, 0.0, 0.0, K)
        Bz0 = apply_currents_general(B0, IyIz, K)
    print_coil_step("Zero By,Bz", out, Bz0, IyIz)

    # 3) Set in-plane B_perp = 12 G at θ=35° from +y towards +z
    By_tgt, Bz_tgt = set_Bperp_theta(B0, Bperp_G=12.0, theta_deg=35.0)
    if K is None:
        IyIz = currents_for_ByBz_targets_diag(B0, By_tgt, Bz_tgt, ky, kz)
        Bp = apply_currents_yz_diag(B0, IyIz[0], IyIz[1], ky, kz)
    else:
        IyIz = currents_for_ByBz_targets_general(B0, By_tgt, Bz_tgt, K)
        Bp = apply_currents_general(B0, IyIz, K)
    print_coil_step("Set B_perp=12G @ 35°", out, Bp, IyIz)

    # 4) Nudge splitting of a chosen NV axis (e.g., axis with lowest f_-)
    j = int(np.argmin(out["f_minus_nvaxes_GHz"]))  # largest |n·B|
    nudge = nudge_axis_split_yz(B0, j, delta_f_MHz=5.0, K_crys_3x2=K, ky_G_per_A=ky, kz_G_per_A=kz)
    print_coil_step(f"Increase splitting on NV {NV_LABELS[j]} by 5 MHz", out, nudge["B_new_G"], nudge["delta_Iyz_A"])

    # 5) Target two axes’ f_- simultaneously (least-squares)
    axes = [0, 1]  # e.g., [ [1,1,1], [-1,1,1] ]
    f_targets = [out["f_minus_nvaxes_GHz"][0] - 3e-3,   # -3 MHz (increase splitting)
                 out["f_minus_nvaxes_GHz"][1] - 2e-3]   # -2 MHz
    two = target_two_axes_splits_yz(B0, axes, f_targets, K_crys_3x2=K, ky_G_per_A=ky, kz_G_per_A=kz)
    print_coil_step(f"Target two axes {NV_LABELS[axes[0]]}, {NV_LABELS[axes[1]]}", out, two["B_new_G"], two["delta_Iyz_A"])
