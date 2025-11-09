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

# ------------- Example -------------
if __name__ == "__main__":
    # Quartets (GHz), internal order doesn't matter for the solver you’re calling
    f00 = [2.7666, 2.7851, 2.8222, 2.8406]     # (0,0)
    f10 = [2.7457, 2.7988, 2.8344, 2.8765]     # (Iy,0), Iy=+1.0
    f01 = [2.7694, 2.8403, 2.7991, 2.8406]     # (0,Iz), Iz=+1.0
    f11 = [2.7457, 2.8170, 2.8100, 2.8751]     # (Iy,Iz), Iy=Iz=+1.0

    out = estimate_K_and_common_offset(f00, f10, f01, f11, Iy=1.0, Iz=1.0)
