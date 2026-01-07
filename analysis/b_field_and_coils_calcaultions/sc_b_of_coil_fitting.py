import numpy as np
from sc_b_field_calculations import solve_B_from_odmr_order_invariant

NV_LABELS = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]

def _solve_B_from_quartet(f4_GHz, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    f4 = np.asarray(f4_GHz, float).reshape(4)
    out = solve_B_from_odmr_order_invariant(f4, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    return np.asarray(out["B"], float)

def estimate_K_and_common_offset(
    f00, f10, f01, f11, Iy=1.0, Iz=1.0, D_GHz=2.8785, gamma=2.8025, verbose=True
):
    """Recover B0, ΔB_common, and K (G/A) from four quartets."""
    # 0) solve B for each quartet (order-invariant)
    B00 = _solve_B_from_quartet(f00, D_GHz, gamma)
    B10 = _solve_B_from_quartet(f10, D_GHz, gamma)  # channel-1 on
    B01 = _solve_B_from_quartet(f01, D_GHz, gamma)  # channel-2 on
    B11 = _solve_B_from_quartet(f11, D_GHz, gamma)  # both on

    Iy = float(Iy); Iz = float(Iz)
    if abs(Iy) < 1e-9 or abs(Iz) < 1e-9:
        raise ValueError("Iy and Iz must be non-zero for this closed-form solve.")

    # 1) common offset that appears whenever coils are energized
    dB_common = (B10 + B01 - B11 - B00)

    # 2) coil-response columns (per amp)
    k1 = (B10 - B00 - dB_common) / Iy
    k2 = (B01 - B00 - dB_common) / Iz
    K  = np.column_stack([k1, k2])  # shape (3,2)

    # 3) reconstruction checks
    B10_hat = B00 + dB_common + K @ np.array([Iy, 0.0])
    B01_hat = B00 + dB_common + K @ np.array([0.0, Iz])
    B11_hat = B00 + dB_common + K @ np.array([Iy, Iz])
    e10 = np.linalg.norm(B10_hat - B10)
    e01 = np.linalg.norm(B01_hat - B01)
    e11 = np.linalg.norm(B11_hat - B11)

    if verbose:
        print("\n=== Baseline B0 (from f00) ===")
        print("B0 (G):", np.round(B00, 6))

        print("\n=== Common coil-on offset (applies when any coil is ON) ===")
        print("ΔB_common (G):", np.round(dB_common, 6))

        print("\n=== Coil→Field map K (crystal frame, G/A) ===")
        print("dB/dI_ch1:", np.round(K[:, 0], 6))
        print("dB/dI_ch2:", np.round(K[:, 1], 6))
        print("  (These are the actual ΔB directions for +1 A on each channel.)")

        print("\n=== Reconstruction errors (Gauss) ===")
        print("||B10_hat - B10_meas|| :", e10)
        print("||B01_hat - B01_meas|| :", e01)
        print("||B11_hat - B11_meas|| :", e11)

        # print("\nWhat is being printed:")
        # print("- 'B0 (G)': baseline field from the (0,0) quartet.")
        # print("- 'ΔB_common (G)': one offset vector present in all coil-on shots (session drift / coil bias).")
        # print("- 'dB/dI_ch1', 'dB/dI_ch2': the coil→field map columns (K) in crystal axes, units G/A.")
        # print("- Reconstruction errors: how well the linear model with ΔB_common reproduces the three coil shots.")

    return {
        "B0_G": B00,
        "K_G_per_A": K,
        "dB_common_G": dB_common,
        "B10_G": B10,
        "B01_G": B01,
        "B11_G": B11,
        "errs_G": (e10, e01, e11),
    }

def fit_B0_K_dBcommon_from_quartets(
    f_list,
    I_list,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    verbose=True,
):
    """
    Least-squares fit of:

        B_meas = B0  +  s * dB_common  +  K @ [I_ch1, I_ch2]

    for many ODMR quartets.

    Inputs
    ------
    f_list : list of length N
        Each element is a 4-element iterable of f_- (GHz) for one shot.
    I_list : list of length N
        Each element is (I_ch1, I_ch2) in Amps for that shot.
    D_GHz, gamma_e_MHz_per_G : NV parameters.

    Model
    -----
    - B0 (3-vector): baseline field with both coils truly OFF.
    - dB_common (3-vector): extra offset whenever any coil is ON
      (e.g. coil magnetization / bias).
    - K (3x2): columns are dB/dI_ch1, dB/dI_ch2 in G/A.

    Returns
    -------
    dict with keys:
      "B0_G", "K_G_per_A", "dB_common_G", "B_meas_G", "B_fit_G", "residuals_G"
    """
    f_list = list(f_list)
    I_list = np.asarray(I_list, float)
    if len(f_list) != len(I_list):
        raise ValueError("f_list and I_list must have the same length.")

    N = len(f_list)

    # 1) Solve B vector from each quartet (order-invariant)
    B_meas = np.empty((N, 3), float)
    for n, f4 in enumerate(f_list):
        B_meas[n, :] = _solve_B_from_quartet(f4, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    # 2) Build design matrix A for each shot:
    #    B_meas = [1, s, I_ch1, I_ch2] @ [B0, dB_common, k1, k2]^T  component-wise
    I1 = I_list[:, 0]
    I2 = I_list[:, 1]
    # coil-on flag: 0 only for (0,0), 1 otherwise (you can customize this)
    s_flag = ((np.abs(I1) > 1e-9) | (np.abs(I2) > 1e-9)).astype(float)

    A = np.column_stack([
        np.ones(N, float),   # B0
        s_flag,              # dB_common
        I1,                  # k1 * I_ch1
        I2,                  # k2 * I_ch2
    ])                       # shape (N, 4)

    # 3) Solve separately for x,y,z components by least squares
    B0 = np.zeros(3, float)
    dB_common = np.zeros(3, float)
    K = np.zeros((3, 2), float)

    for comp in range(3):
        b = B_meas[:, comp]       # shape (N,)
        # coefficients: [B0_comp, dB_common_comp, k1_comp, k2_comp]
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        B0[comp] = coeffs[0]
        dB_common[comp] = coeffs[1]
        K[comp, 0] = coeffs[2]
        K[comp, 1] = coeffs[3]

    # 4) Reconstruct fitted fields and residuals
    B_fit = np.empty_like(B_meas)
    for n in range(N):
        s = s_flag[n]
        I_vec = np.array([I1[n], I2[n]], float)
        B_fit[n, :] = B0 + s * dB_common + K @ I_vec

    residuals = B_fit - B_meas
    rms_res = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

    if verbose:
        print("\n=== LS fit: B0, dB_common, K ===")
        print("B0 (G):", np.round(B0, 6))
        print("ΔB_common (G):", np.round(dB_common, 6))
        print("dB/dI_ch1 (G/A):", np.round(K[:, 0], 6))
        print("dB/dI_ch2 (G/A):", np.round(K[:, 1], 6))
        print("RMS residual (G):", rms_res)

    return {
        "B0_G": B0,
        "K_G_per_A": K,
        "dB_common_G": dB_common,
        "B_meas_G": B_meas,
        "B_fit_G": B_fit,
        "residuals_G": residuals,
        "rms_residual_G": rms_res,
    }
def infer_effective_currents_from_quartet(
    f4_GHz,
    B0_G,
    K_G_per_A,
    dB_common_G,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
):
    """
    Given one quartet and a trusted calibration (B0, K, dB_common),
    solve for the effective currents I_eff = (I1, I2) that would
    produce the measured B within that linear model.
    """
    # field from the quartet
    B_meas = _solve_B_from_quartet(f4_GHz, D_GHz=D_GHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)

    # target delta B
    dB = B_meas - B0_G - dB_common_G

    # least-squares solve K * I_eff = dB
    I_eff, *_ = np.linalg.lstsq(K_G_per_A, dB, rcond=None)
    return I_eff, B_meas

import numpy as np
from typing import List, Tuple, Dict

from sc_b_field_calculations import solve_B_from_odmr_order_invariant  # or adjust import


def fit_B0_K_dBcommon_from_quartets(
    f_list: List[np.ndarray],
    I_list: List[Tuple[float, float]],
    D_GHz: float = 2.8785,
    gamma_e_MHz_per_G: float = 2.8025,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Global LS fit of:
        B_k ≈ B0 + s_k * dB_common + K @ [I1_k, I2_k]
    using multiple quartets measured at known currents (I1_k, I2_k).

    Inputs
    ------
    f_list : list of quartets, one per shot
        Each element is an iterable of 4 f_- frequencies (GHz), any internal order.
    I_list : list of (I1, I2) tuples, same length as f_list
        Coil currents in Amps.
    D_GHz, gamma_e_MHz_per_G : NV parameters for the order-invariant solver.

    Returns
    -------
    dict with:
        "B0_G"        : baseline field (3,)
        "K_G_per_A"   : coil map (3,2)  [columns = dB/dI_ch1, dB/dI_ch2]
        "dB_common_G" : coil-on offset (3,)
        "B_meas_G"    : array (N,3) of measured fields from quartets
        "B_fit_G"     : array (N,3) of LS-predicted fields
        "residuals_G" : array (N,) of |B_fit - B_meas|
    """
    if len(f_list) != len(I_list):
        raise ValueError("f_list and I_list must have the same length.")

    N = len(f_list)
    B_meas = np.zeros((N, 3), float)
    I_arr  = np.zeros((N, 2), float)
    s_arr  = np.zeros(N, float)

    # 1) Convert all quartets -> B_meas via your order-invariant solver
    for k, (f4, (I1, I2)) in enumerate(zip(f_list, I_list)):
        I1 = float(I1); I2 = float(I2)
        I_arr[k, :] = (I1, I2)
        s_arr[k] = 0.0 if (abs(I1) < 1e-12 and abs(I2) < 1e-12) else 1.0

        out = solve_B_from_odmr_order_invariant(
            f4,
            D_GHz=D_GHz,
            gamma_e_MHz_per_G=gamma_e_MHz_per_G,
        )
        B_meas[k, :] = np.asarray(out["B"], float)

    # 2) Build LS system A p ≈ y.
    #
    # Parameter vector p has 12 entries:
    #   [B0_x, B0_y, B0_z,
    #    dBc_x, dBc_y, dBc_z,
    #    K11, K12,   # x-component: dB_x/dI1, dB_x/dI2
    #    K21, K22,   # y-component
    #    K31, K32]   # z-component
    #
    # For each shot k and component c in {0,1,2}:
    #   B_meas[k,c] ≈ B0_c + s_k * dBc_c + Kc1 * I1_k + Kc2 * I2_k

    n_params = 12
    A = np.zeros((3 * N, n_params), float)
    y = np.zeros(3 * N, float)

    for k in range(N):
        I1, I2 = I_arr[k, :]
        s = s_arr[k]
        for c in range(3):
            row = 3 * k + c
            y[row] = B_meas[k, c]

            # B0_c
            A[row, c] = 1.0
            # dB_common_c (offset when coils on)
            A[row, 3 + c] = s
            # Kc1, Kc2
            #   x: K11,K12 are indices 6,7
            #   y: K21,K22 are indices 8,9
            #   z: K31,K32 are indices 10,11
            K1_idx = 6 + 2 * c
            K2_idx = 7 + 2 * c
            A[row, K1_idx] = I1
            A[row, K2_idx] = I2

    # 3) Solve least squares
    p, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)

    # Unpack parameters
    B0 = p[0:3]
    dB_common = p[3:6]
    K = np.zeros((3, 2), float)
    K[0, :] = p[6:8]
    K[1, :] = p[8:10]
    K[2, :] = p[10:12]

    # 4) Reconstruct B_fit for each shot; compute residuals
    B_fit = np.zeros_like(B_meas)
    res_norms = np.zeros(N, float)
    for k in range(N):
        I1, I2 = I_arr[k, :]
        s = s_arr[k]
        B_fit[k, :] = B0 + s * dB_common + K @ np.array([I1, I2], float)
        res_norms[k] = np.linalg.norm(B_fit[k, :] - B_meas[k, :])

    if verbose:
        print("\n=== LS fit: B0, dB_common, K ===")
        print("B0 (G):", np.round(B0, 6))
        print("ΔB_common (G):", np.round(dB_common, 6))
        print("dB/dI_ch1 (G/A):", np.round(K[:, 0], 6))
        print("dB/dI_ch2 (G/A):", np.round(K[:, 1], 6))
        print("RMS residual (G):", np.sqrt(np.mean(res_norms**2)))

        print("\nPer-shot residuals (G):")
        for k, (f4, (I1, I2)) in enumerate(zip(f_list, I_list)):
            print(
                f"shot {k:2d} I=({I1:+.2f}, {I2:+.2f})  "
                f"B_meas={np.round(B_meas[k, :], 3)}  "
                f"B_fit={np.round(B_fit[k, :], 3)}  "
                f"|ΔB|={res_norms[k]:5.2f} G"
            )

    return {
        "B0_G": B0,
        "K_G_per_A": K,
        "dB_common_G": dB_common,
        "B_meas_G": B_meas,
        "B_fit_G": B_fit,
        "residuals_G": res_norms,
    }

if __name__ == "__main__":
    # Quartets (GHz), order doesn't matter for the solver
    f00 = [2.7666, 2.7851, 2.8222, 2.8406]     # (0,0)
    f10 = [2.7457, 2.7988, 2.8344, 2.8765]     # (1,0)
    f01 = [2.7694, 2.8403, 2.7991, 2.8406]     # (0,1)
    f11 = [2.7457, 2.8170, 2.8100, 2.8751]     # (1,1)
    f3m3 = [2.7245, 2.7471,  2.8480, 2.8282]   # (3,-3)
    f33  = [2.7058, 2.8699,  2.7859, 2.8300]   # (3, 3)

    f_list = [f00, f10, f01, f11, f3m3, f33]
    I_list = [
        (0.0, 0.0),   # f00
        (1.0, 0.0),   # f10
        (0.0, 1.0),   # f01
        (1.0, 1.0),   # f11
        (3.0,-3.0),   # f3m3
        (3.0, 3.0),   # f33
    ]

    fit_out = fit_B0_K_dBcommon_from_quartets(
        f_list,
        I_list,
        D_GHz=2.8785,
        gamma_e_MHz_per_G=2.8025,
        verbose=True,
    )

    B0_new = fit_out["B0_G"]
    K_new  = fit_out["K_G_per_A"]
    dB_common_new = fit_out["dB_common_G"]

    print("\n=== Use these values in CoilField3D if you like ===")
    print("B0_DEFAULT =", np.round(B0_new, 6))
    print("K_DEFAULT  = np.column_stack([",
          np.round(K_new[:,0], 6), ",",
          np.round(K_new[:,1], 6), "])")


# # ------------- Example -------------
# if __name__ == "__main__":
#     # Quartets (GHz), internal order doesn't matter for the solver you’re calling
#     f00 = [2.7666, 2.7851, 2.8222, 2.8406]     # (0,0)
#     f10 = [2.7457, 2.7988, 2.8344, 2.8765]     # (Iy,0), Iy=+1.0
#     f01 = [2.7694, 2.8403, 2.7991, 2.8406]     # (0,Iz), Iz=+1.0
#     f11 = [2.7457, 2.8170, 2.8100, 2.8751]     # (Iy,Iz), Iy=Iz=+1.0
#     f3m3 = [2.7245, 2.7471,  2.8480, 2.8282]
#     f33 = [2.7058,  2.8699, 2.7859, 2.8300]
#     # out = estimate_K_and_common_offset(f00, f10, f01, f11, Iy=1.0, Iz=1.0)

