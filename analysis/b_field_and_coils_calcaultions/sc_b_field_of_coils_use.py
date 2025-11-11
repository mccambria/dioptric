# coil_field_3d.py
# Build the 3D magnetic field vector from two coil currents using your calibrated K,
# sweep grids in current space, and (optionally) predict exact ODMR quartets.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from sc_b_field_of_coils import exact_f_minus_quartet 
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
# --- A) Relabel quartet into NV-axes order given the CURRENT field ---
from itertools import permutations
# ---------- Your locked-in calibration (crystal frame) ----------
B0_DEFAULT = np.array([-46.275577, -17.165999, -5.701398], float)

# K = [dB/dI_ch1, dB/dI_ch2] (columns), in Gauss per Amp
K_DEFAULT = np.column_stack([
    [ +0.803449,  -4.758891, -15.327336],   # dB/dI_ch1
    [ -0.741645,  +4.202657,  -4.635283],   # dB/dI_ch2
]).astype(float)

# ---------- (Optional) exact quartet model hook ----------
# If you already have an exact Hamiltonian solver, import it and set this flag True
USE_EXACT = True
try:
    # Replace with your function if available:
    # from sc_b_field_of_coils import exact_f_minus_quartet
    # def exact_f_minus_quartet(B_crys_G: np.ndarray, D_GHz=2.8785, E_MHz=0.0, gamma_e_MHz_per_G=2.8025) -> np.ndarray: ...
    pass
except Exception:
    USE_EXACT = False

D_GHZ   = 2.8785
GAMMA   = 2.8025  # MHz/G
E_MHZ   = 0.0
    # --- NV labels and axes ---
NV_LABELS = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]

def _nv_axes():
    a = np.array([[ 1,  1,  1],
                [-1,  1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1]], float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a  # (4,3)


def relabel_quartet_to_nv_order_exact(B_crys_G, f4_any_order_GHz,
                                      D_GHz, E_MHz, gamma_e_MHz_per_G):
    """
    Return (f4_in_nv_order, perm, err_MHz2) using exact model matching.

    We compute the predicted quartet in canonical NV order for this B.
    Then we find the permutation of the measured 4 peaks that minimizes
    sum_j (f_meas[p[j]] - f_pred[j])^2.
    """
    f_pred = exact_f_minus_quartet(
        np.asarray(B_crys_G, float).reshape(3),
        D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G
    )
    f_meas = np.asarray(f4_any_order_GHz, float).reshape(4)

    best_perm, best_err = None, np.inf
    for p in permutations(range(4)):
        err = np.sum((1000.0*(f_meas[list(p)] - f_pred))**2)   # MHz^2
        if err < best_err:
            best_err, best_perm = err, p

    f_nv = np.empty(4, float)
    for j_in, i_nv in enumerate(best_perm):    # assign into NV order
        f_nv[j_in] = f_meas[i_nv]
    return f_nv, best_perm, best_err


# --- B) Optional: retune D to anchor exact peaks to your baseline quartet ---
def retune_D_to_baseline(f00_meas_GHz, B0_G, D_GHz, E_MHz, gamma_e_MHz_per_G):
    """
    Shift D so the mean predicted baseline f_- equals the mean measured f_-.
    Good first-order correction for uniform offset (~few MHz).
    """
    f_pred = exact_f_minus_quartet(
        B0_G, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G
    )
    delta_MHz = 1000.0 * (np.mean(f00_meas_GHz) - np.mean(f_pred))
    return D_GHz + delta_MHz/1000.0

def print_residual_table(B, f_meas_any_order, D_GHz, E_MHz, gamma_e_MHz_per_G):
    f_pred = exact_f_minus_quartet(B, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
    f_meas_nv, perm, err = relabel_quartet_to_nv_order_exact(B, f_meas_any_order, D_GHz, E_MHz, gamma_e_MHz_per_G)
    nv_labels = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]
    print("perm (meas->NV):", perm, "  RMS Δf (MHz):", (err/4)**0.5)
    for lab, fm, fp in zip(nv_labels, f_meas_nv, f_pred):
        print(f"{lab:>9}   f_meas={fm:.9f}  f_pred={fp:.9f}   Δf={(1000*(fm-fp)):+.3f} MHz")

@dataclass
class CoilField3D:
    """Map coil currents (I_ch1, I_ch2) -> 3D magnetic field in crystal frame."""
    B0: np.ndarray = B0_DEFAULT.copy()
    K:  np.ndarray = K_DEFAULT.copy()

    def field_from_currents(self, I_ch1: float, I_ch2: float, drift: Optional[np.ndarray]=None
                            ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Return (B, |B|, B_hat) for given currents (A). Optional 'drift' (G) is added if provided.
        """
        I = np.array([float(I_ch1), float(I_ch2)], float)
        B = self.B0 + self.K @ I
        if drift is not None:
            B = B + np.asarray(drift, float).reshape(3)
        Bmag = float(np.linalg.norm(B))
        Bhat = (B/Bmag) if Bmag > 0 else np.array([0.0, 0.0, 1.0])
        return B, Bmag, Bhat

    def field_grid(self, I1_range: Tuple[float,float], I2_range: Tuple[float,float],
                   n1: int=41, n2: int=41, drift: Optional[np.ndarray]=None
                   ) -> Dict[str, np.ndarray]:
        """
        Sweep a rectangular grid in current space and return arrays:
        Iy_grid, Iz_grid, Bx, By, Bz, Bmag  (all shapes (n1,n2)).
        """
        i1 = np.linspace(I1_range[0], I1_range[1], int(n1))
        i2 = np.linspace(I2_range[0], I2_range[1], int(n2))
        Iy, Iz = np.meshgrid(i1, i2, indexing="ij")  # (n1,n2)

        # Vectorized: B = B0[:,None,None] + K @ [Iy, Iz]
        col1 = self.K[:, 0].reshape(3,1,1) * Iy[None,:,:]
        col2 = self.K[:, 1].reshape(3,1,1) * Iz[None,:,:]
        B = self.B0.reshape(3,1,1) + col1 + col2
        if drift is not None:
            B = B + np.asarray(drift, float).reshape(3,1,1)

        Bx, By, Bz = B[0], B[1], B[2]
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        return {"Iy_grid": Iy, "Iz_grid": Iz, "Bx": Bx, "By": By, "Bz": Bz, "Bmag": Bmag}

    def currents_for_deltaB(self, dB_target_G: np.ndarray, drift: Optional[np.ndarray]=None
                            ) -> Dict[str, np.ndarray]:
        """
        Least-squares currents that achieve a target ΔB in the span of K.
        Returns Iy, Iz, ΔB_achieved, residual.
        """
        dB = np.asarray(dB_target_G, float).reshape(3)
        if drift is not None:
            dB = dB - np.asarray(drift, float).reshape(3)
        dI, *_ = np.linalg.lstsq(self.K, dB, rcond=None)
        dB_ach = self.K @ dI
        return {
            "Iy_A": float(dI[0]), "Iz_A": float(dI[1]),
            "dB_achieved_G": dB_ach, "dB_residual_G": dB_ach - dB
        }

    def currents_for_target_B(self, B_target_G: np.ndarray, drift: Optional[np.ndarray]=None
                              ) -> Dict[str, np.ndarray]:
        """
        Minimal-norm LS currents to move from B0 to a specified B_target (both crystal frame).
        """
        B_target = np.asarray(B_target_G, float).reshape(3)
        dB_req = B_target - self.B0
        if drift is not None:
            dB_req = dB_req - np.asarray(drift, float).reshape(3)
        return self.currents_for_deltaB(dB_req)

    def peaks_exact(self, B_crys_G: np.ndarray,
                    D_GHz: float=D_GHZ, E_MHz: float=E_MHZ, gamma_e_MHz_per_G: float=GAMMA
                    ) -> Optional[np.ndarray]:
        """
        Optional: return the exact quartet (GHz) if your exact solver is available.
        """
        if not USE_EXACT:
            return None
        return exact_f_minus_quartet(np.asarray(B_crys_G, float).reshape(3),
                                     D_GHz=D_GHz, E_MHz=E_MHz,
                                     gamma_e_MHz_per_G=gamma_e_MHz_per_G)


    def predict_peaks(self,
                    I_ch1: float,
                    I_ch2: float,
                    drift: Optional[np.ndarray] = None,
                    D_GHz: float = D_GHZ,
                    E_MHz: float = E_MHZ,
                    gamma_e_MHz_per_G: float = GAMMA) -> Dict[str, np.ndarray]:
        """
        Currents -> field -> exact ODMR quartet (GHz) in NV-axes order:
        [ [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1] ]
        """
        B, Bmag, Bhat = self.field_from_currents(I_ch1, I_ch2, drift=drift)
        f4_nv = None
        perm = None
        err = None
        if USE_EXACT:
            f4_raw = exact_f_minus_quartet(B, D_GHz=D_GHz, E_MHz=E_MHz, gamma_e_MHz_per_G=gamma_e_MHz_per_G)
            # reorder measured/“raw” peaks into NV order by matching to exact-predicted NV-order peaks
            f4_nv, perm, err = relabel_quartet_to_nv_order_exact(B, f4_raw, D_GHz, E_MHz, gamma_e_MHz_per_G)

        return {"B_G": B, "|B|_G": Bmag, "B_hat": Bhat,
                "f_minus_GHz": f4_nv, "raw_peaks_GHz": (f4_raw if USE_EXACT else None),
                "perm_input_to_NV": perm, "relabel_error": err}


    def field_maps_over_currents(cal,
                                Iy_range=(-4.0, 4.0),
                                Iz_range=(-4.0, 4.0),
                                nI=201,
                                target_Bmag_G=None,
                                use_baseline =True,
                                annotate_points=()):
        """
        2-D maps of Bx, By, Bz, and |B| vs (Iy, Iz).

        Params
        ------
        cal : your NVCalib (must expose cal.field_from_currents(iy, iz) -> (B_vec, |B|, extras))
        Iy_range, Iz_range : (min, max) in Amps for the two channels
        nI : number of samples per axis
        target_Bmag_G : if given, draw a |B|=target contour
        annotate_points : list of (Iy, Iz, label) to mark on all maps
        """
        Iy = np.linspace(Iy_range[0], Iy_range[1], nI)
        Iz = np.linspace(Iz_range[0], Iz_range[1], nI)
        IZZ, IYY = np.meshgrid(Iz, Iy)  # note: X=Iz, Y=Iy for plotting

        # compute fields on the grid
        Bx = np.empty_like(IZZ, dtype=float)
        By = np.empty_like(IZZ, dtype=float)
        Bz = np.empty_like(IZZ, dtype=float)
        Bm = np.empty_like(IZZ, dtype=float)

        # for i in range(nI):
        #     for j in range(nI):
        #         B, M, _ = cal.field_from_currents(IYY[i, j], IZZ[i, j])
        #         Bx[i, j], By[i, j], Bz[i, j] = B
        #         Bm[i, j] = M
        
        # choose baseline mode
        B0_used = cal.B0 if use_baseline else np.zeros(3)

        for i in range(nI):
            for j in range(nI):
                # compute field with selected baseline
                I = np.array([IYY[i, j], IZZ[i, j]], float)
                B = B0_used + cal.K @ I
                M = float(np.linalg.norm(B))
                Bx[i, j], By[i, j], Bz[i, j] = B
                Bm[i, j] = M

        # helper to make lots of contours
        def contour_levels(arr, major=5.0, minor=1.0, symmetric=False):
            a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
            if symmetric:
                m = max(abs(a_min), abs(a_max))
                a_min, a_max = -m, m
            # major & minor grids
            maj = np.arange(np.floor(a_min/major)*major, np.ceil(a_max/major)*major + 0.5*major, major)
            mnr = np.arange(np.floor(a_min/minor)*minor, np.ceil(a_max/minor)*minor + 0.5*minor, minor)
            return maj, mnr

        # 2x2 maps
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        panels = [
            ("Bx (G)", Bx, True),
            ("By (G)", By, True),
            ("Bz (G)", Bz, True),
            ("|B| (G)", Bm, False),
        ]

        for ax, (title, Z, symm) in zip(axs.flat, panels):
            maj, mnr = contour_levels(Z, major=5.0, minor=1.0, symmetric=symm)
            # filled minor contours + line major contours
            cf = ax.contourf(IZZ, IYY, Z, levels=mnr, extend="both")
            cs_maj = ax.contour(IZZ, IYY, Z, levels=maj, colors="k", linewidths=1.0)
            ax.clabel(cs_maj, fmt="%.0f", inline=True, fontsize=8)
            # zero contour (for Bx,By,Bz)
            if symm:
                ax.contour(IZZ, IYY, Z, levels=[0.0], colors="white", linewidths=2.0, linestyles="--", alpha=0.9)

            # optional |B| target on every panel for reference
            if target_Bmag_G is not None and title != "Bx (G)":  # draw on By, Bz, |B|
                cs_t = ax.contour(IZZ, IYY, Bm, levels=[float(target_Bmag_G)], colors="red", linewidths=2.0)
                try:
                    ax.clabel(cs_t, fmt=f"|B|={float(target_Bmag_G):.0f} G", inline=True, fontsize=9)
                except Exception:
                    pass

            # annotate points
            # for (iy0, iz0, lab) in annotate_points:
            #     ax.plot(iz0, iy0, 'o', ms=6, mec='k', mfc='yellow', zorder=4)
            #     if lab:
            #         ax.text(iz0, iy0, f" {lab}", va="center", ha="left", fontsize=8, color="k")

            ax.set_title(title)
            ax.set_xlabel("I_ch2 (A)")
            ax.set_ylabel("I_ch1 (A)")
            ax.grid(alpha=0.2)
            cbar = fig.colorbar(cf, ax=ax)
            cbar.ax.set_ylabel(title)

        mode = "Coils + Magnet" if use_baseline else "Coils Only"
        fig.suptitle(
            f"Bx, By, Bz, |B| vs (Iy, Iz) — {mode} (Crystal Frame)\n"
            f"|B0|={np.linalg.norm(B0_used):.1f} G"
        )
        plt.show()


# -------------------------- Examples --------------------------
if __name__ == "__main__":
    kpl.init_kplotlib()
    cal = CoilField3D()

    # # (1) Single prediction
    # Iy, Iz = 0.73, 1.54
    # B, Bmag, Bhat = cal.field_from_currents(Iy, Iz)
    # print("=== Single prediction ===")
    # print(f"I_ch1={Iy:.3f} A, I_ch2={Iz:.3f} A")
    # print("B (G):   ", np.round(B, 6))
    # print("|B| (G): ", f"{Bmag:.6f}")
    # print("B̂:       ", np.round(Bhat, 6))

    # # (2) Currents to reach a target B
    # B_target = np.array([-40.0, -18.0, -10.0])
    # plan = cal.currents_for_target_B(B_target)
    # print("\n=== Currents for target B ===")
    # print("B_target (G):    ", np.round(B_target, 6))
    # print("Iy (A), Iz (A):  ", plan["Iy_A"], plan["Iz_A"])
    # B2, B2mag, _ = cal.field_from_currents(plan["Iy_A"], plan["Iz_A"])
    # print("B_achieved (G):  ", np.round(B2, 6), "  |B|=", f"{B2mag:.6f} G")
    # print("ΔB residual (G): ", np.round(plan["dB_residual_G"], 6))

    # # (3) Build a current grid and get full 3D field arrays
    # grid = cal.field_grid(I1_range=(-2.0, 2.0), I2_range=(-2.0, 2.0), n1=51, n2=51)
    # print("\n=== Grid summary ===")
    # print("Iy_grid shape:", grid["Iy_grid"].shape)
    # print("Bmag stats (G): min={:.3f}, max={:.3f}".format(grid["Bmag"].min(), grid["Bmag"].max()))


    # (A) Predict quartet at a chosen operating point
    # Iy, Iz = 0.73, 1.54
    Iy, Iz = 1.0, 1.0
    out = cal.predict_peaks(Iy, Iz)
    print("\n=== Peak prediction ===")
    print(f"Inputs: I_ch1 = {Iy:.6f} A, I_ch2 = {Iz:.6f} A")
    print("Field (crystal frame):")
    print(" B (G):   ", np.round(out["B_G"], 6))
    print(" |B| (G): ", f"{out['|B|_G']:.6f}")
    print(" B_hat:   ", np.round(out["B_hat"], 6))

    if out["f_minus_GHz"] is not None:
        nv_labels = ["[1,1,1]", "[-1,1,1]", "[1,-1,1]", "[1,1,-1]"]
        print("\nQuartet f_- (GHz) in NV-axes order:")
        for lab, val in zip(nv_labels, out["f_minus_GHz"]):
            print(f" {lab:<9} {val:.9f}")
    else:
        print("\n(Set USE_EXACT=True to print exact peaks.)")
    
    print("perm (raw->NV order):", out["perm_input_to_NV"], " map_err:", out["relabel_error"])

    # out = cal.predict_peaks(0.0, 1.0)
    # B = out["B_G"]
    # f_meas_any_order = [...] 
    # print_residual_table(B, f_meas_any_order, D_GHZ, E_MHZ, GAMMA)
    # # ----------------------------
    # # Figure: |B| contour with the (Iy, Iz) path
    # # ----------------------------
    # Pick your grid & an example target
    Iy_rng = (-4.0, 4.0)
    Iz_rng = (-4.0, 4.0)
    target = 60.0  # draw the |B|=60 G contour

    # Optionally mark a few operating points
    marks = [
        (0.00, 0.00, "baseline"),
        (0.73, 1.54, "op pt"),
        (1.00, 0.00, "ch1=1A"),
        (0.00, 1.00, "ch2=1A"),
    ]

    # cal.field_maps_over_currents(Iy_range=Iy_rng, Iz_range=Iz_rng, nI=161,
    #                         target_Bmag_G=target, use_baseline=False, annotate_points=marks)


    kpl.show(block=True)