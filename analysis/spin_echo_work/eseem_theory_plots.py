
# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline

Author: @saroj chand
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import data_manager as dm
from utils import kplotlib as kpl
from analysis.sc_c13_hyperfine_sim_data_driven import (
    read_hyperfine_table_safe,
    B_vec_T,   # your lab field (Tesla)
    gamma_C13,
    make_R_NV,
)
from analysis.spin_echo_work.echo_fit_models import fine_decay, fine_decay_fixed_revival

def check_axis(ax):
    R = make_R_NV(ax)
    # 1) Proper rotation?
    print(ax, "det(R)=", np.linalg.det(R))  # ~ +1
    print("||R R^T - I||_F =", np.linalg.norm(R @ R.T - np.eye(3)))  # ~ 0

    # 2) R maps the NV axis to ez (NV frame)
    n = np.asarray(ax, float) / np.linalg.norm(ax)
    ez_from_R = R @ n
    print("R @ n =", ez_from_R)  # ~ [0,0,1]

    # 3) Rotate B into NV frame and normalize
    B_NV = R @ B_vec_T
    B_hat = B_NV / np.linalg.norm(B_NV)
    print("B_hat_NV =", B_hat, "||B_hat||=", np.linalg.norm(B_hat))

    # 4) Angle between B and NV axis (in NV frame this is arccos of z-component)
    cos_theta = B_hat[2]
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    print("theta(B, NV-axis) [deg] =", theta_deg, "\n")


# Pauli/2 operators in Hz (dimensionless matrices; coefficients carry Hz units)
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=float)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=float)

def _build_U_from_orientation(orientation, phi_deg=0.0):
    """Columns of U are NV-frame basis expressed in cubic coords:
       ex, ey in the NV plane; ez along NV axis (orientation)."""
    ez = np.asarray(orientation, float); ez /= np.linalg.norm(ez)
    trial = np.array([1.0, -1.0, 0.0])
    if abs(np.dot(trial/np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0, 1.0, -1.0])
    ex = trial - np.dot(trial, ez)*ez; ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    U0 = np.column_stack([ex, ey, ez])
    phi = np.deg2rad(phi_deg)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                   [np.sin(phi),  np.cos(phi), 0.0],
                   [0.0,          0.0,        1.0]])
    return U0 @ Rz, ez  # U maps NV-frame tensors -> cubic; ez is NV axis in cubic


def essem_lines_by_diag(
    A_file_Hz: np.ndarray,
    orientation=(1,1,1),
    B_lab_vec=None,                 # e.g. in Tesla; pass gamma accordingly
    gamma_n_Hz_per_T=10.705e6, # 13C: 10.705 MHz/T. If B in G, use 10705 Hz/G
    ms=-1,
    phi_deg=0.0,
):
    """
    Returns: f_minus_Hz, f_plus_Hz, fI_Hz, omega_ms_Hz
    Diagonalizes the I=1/2 nuclear Hamiltonian in ms=0 and ms=±1 manifolds.
    """
    # 1) Map A (NV frame) -> cubic; get NV axis in cubic
    U, z_nv_cubic = _build_U_from_orientation(orientation, phi_deg=phi_deg)
    A_cubic = U @ A_file_Hz @ U.T  # Hz

    # 2) Nuclear Zeeman in cubic frame
    B_lab = np.asarray(B_lab_vec, float)
    Bmag = float(np.linalg.norm(B_lab))
    if Bmag == 0.0:
        raise ValueError("B field magnitude is zero.")
    bx, by, bz = (B_lab / Bmag)  # unit vector
    fI_Hz = gamma_n_Hz_per_T * Bmag

    # Zeeman Hamiltonian H_Z = fI * (b·σ)/2  (in Hz units)
    HZ = fI_Hz * (bx * Sx + by * Sy + bz * Sz)

    # 3) Hyperfine effective field felt by nucleus in ms manifold:
    #    H_hf = ms * (A_cubic @ z_nv) · I   (I ~ σ/2)
    Aeff_vec = A_cubic @ z_nv_cubic  # Hz-vector
    Hhf = float(ms) * (Aeff_vec[0]*Sx + Aeff_vec[1]*Sy + Aeff_vec[2]*Sz)

    # 4) Manifold Hamiltonians and splittings (eigenvalue difference)
    H0   = HZ                         # ms=0
    Hms  = HZ + Hhf                   # ms=±1 (default -1)

    evals0 = np.linalg.eigvalsh(H0)
    evalsms = np.linalg.eigvalsh(Hms)

    fI_split     = float(np.abs(evals0[1]  - evals0[0]))   # = fI_Hz (sanity)
    omega_ms_split = float(np.abs(evalsms[1] - evalsms[0]))

    # 5) ESEEM combination lines
    f_minus = abs(omega_ms_split - fI_split)
    f_plus  =      omega_ms_split + fI_split

    return f_minus, f_plus, fI_split, omega_ms_split


# 1) file (NV) -> cubic, but the mapping MUST depend on the NV orientation
def A_file_to_cubic_for_orientation(A_file, orientation, phi_deg=0.0):
    # ez along the NV axis (the given orientation in cubic)
    ez = np.asarray(orientation, float)
    ez /= np.linalg.norm(ez)

    # pick an ex perpendicular to ez; start from [1,-1,0] and project out ez
    trial = np.array([1.0, -1.0, 0.0])
    # if nearly collinear, choose a different trial
    if abs(np.dot(trial / np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0, 1.0, -1.0])

    ex = trial - np.dot(trial, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    U0 = np.column_stack([ex, ey, ez])  # columns are NV-file axes in cubic coords

    # Optional azimuthal twist around ez to match the table’s in-plane choice
    phi = np.deg2rad(phi_deg)
    Rz = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    U = U0 @ Rz

    # map A_file (NV frame) -> cubic
    return U @ A_file @ U.T

def plot_sorted_hyperfine_and_essem(
    hyperfine_path: str,
    orientation=(1, 1, 1),
    distance_max: float = 22.0,  # Å
    title_suffix: str = "",
    project: str = "B",  # "B" (recommended) or "NV"
    file_frame: str = "111",  # "111" if your file is z||<111>, else "cubic"
    freq_range_khz = (10, 10000),
):
    # --- Load & prune ---
    df = read_hyperfine_table_safe(hyperfine_path)
    df = df[df["distance"] <= float(distance_max)].copy()

    MIN_KHZ, MAX_KHZ = freq_range_khz 
    def _in_range(arr, lo=MIN_KHZ, hi=MAX_KHZ):
        a = np.asarray(arr, float)
        m = (a >= lo) & (a <= hi) & np.isfinite(a)
        return a[m], m
    # --- NV rotation & B in this NV frame (do once) ---
    # 0) Prepare B in the lab/cubic frame ONCE
    B_lab = np.asarray(B_vec_T, float)  # keep in cubic
    B_mag = float(np.linalg.norm(B_lab))
    if B_mag == 0.0:
        raise ValueError("B field magnitude is zero.")
    B_hat_cubic = B_lab / B_mag
    f_I_Hz = gamma_C13 * B_mag

    # --- prepare B ONCE in cubic/lab frame ---
    B_lab = np.asarray(B_vec_T, float)
    B_mag = float(np.linalg.norm(B_lab))
    if B_mag == 0.0:
        raise ValueError("B field magnitude is zero.")
    B_hat_cubic = B_lab / B_mag
    f_I_Hz = gamma_C13 * B_mag  # units must match B_lab

    # --- per-site loop ---
    Apar_kHz, Aperp_kHz, fplus_kHz, fminus_kHz = [], [], [], []
    # --- per-site loop ---
    fplus_list  = []
    fminus_list = []
    for _, row in df.iterrows():
        # A_file (MHz -> Hz) in NV(111) frame (the table frame)
        A_file = (
            np.array(
                [
                    [row.Axx, row.Axy, row.Axz],
                    [row.Axy, row.Ayy, row.Ayz],
                    [row.Axz, row.Ayz, row.Azz],
                ],
                float,
            )
            * 1e6
        )

        # file -> cubic, using THIS NV's orientation
        if file_frame.lower() == "111":
            A_cubic = A_file_to_cubic_for_orientation(A_file, orientation, phi_deg=0.0)
        elif file_frame.lower() == "cubic":
            A_cubic = A_file
        else:
            raise ValueError("file_frame must be '111' or 'cubic'")

        # choose projection axis u
        if project.lower() == "b":
            u = B_hat_cubic  # DO NOT rotate B
            A_use = A_cubic  # project tensor in cubic frame onto B
            proj_txt = "proj on B"
        elif project.lower() == "nv":
            u = np.array([0.0, 0.0, 1.0])  # NV axis in NV table frame
            A_use = A_file  # project tensor in NV frame onto NV axis
            proj_txt = "proj on NV"
        else:
            raise ValueError("project must be 'B' or 'NV'")

        # components along u (works for both branches)
        A_par = float(u @ A_use @ u)  # Hz
        A_perp = float(np.linalg.norm(A_use @ u - A_par * u))  # Hz

        # ESEEM frequencies (Hz)  <-- ALWAYS compute (not only in 'nv' branch)
        f_m1 = np.sqrt((f_I_Hz - A_par) ** 2 + A_perp**2)
        f_plus = f_m1 + f_I_Hz
        f_minus = f_m1 - f_I_Hz

        # store magnitudes (kHz)  <-- ALWAYS append
        # Apar_kHz.append(abs(A_par) / 1e3)
        # Aperp_kHz.append(abs(A_perp) / 1e3)
        # fplus_kHz.append(abs(f_plus) / 1e3)
        # fminus_kHz.append(abs(f_minus) / 1e3)

        # A_file in Hz from your table (NV frame)
        f_minus_Hz, f_plus_Hz, fI_Hz, omega_ms_Hz = essem_lines_by_diag(
            A_file_Hz=A_file,
            orientation=orientation,
            B_lab_vec=B_lab,               # same B you already use (cubic)
            gamma_n_Hz_per_T=10.705e6,     # if B in Tesla. If B in Gauss, use 10705.0
            ms=-1,                         # your echo uses 0 <-> -1
            phi_deg=0.0
        )

        # store in kHz if you like:
        fminus_kHz.append(f_minus_Hz/1e3)
        fplus_kHz.append(f_plus_Hz/1e3)
        
        fminus_list.append(f_minus_Hz / 1e3)  # kHz
        fplus_list.append(f_plus_Hz / 1e3)


    # Apar_kHz, mA = _in_range(Apar_kHz)
    # Aperp_kHz, mP = _in_range(Aperp_kHz)
    fplus_kHz, mF = _in_range(fplus_kHz)
    fminus_kHz, mM = _in_range(fminus_kHz)

    theta_deg = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    B_hat_cubic,
                    np.asarray(orientation, float) / np.linalg.norm(orientation),
                ),
                -1,
                1,
            )
        )
    )
    print(
        f"NV {orientation}: angle(B, NV-axis) = {theta_deg:.2f}°  |B|={B_mag:.3g} (units of B_vec_T)"
    )

    # --- sort by magnitude ---
    def _sorted_mag(a):
        a = np.asarray(a, float)
        idx = np.argsort(a)  # already positive
        return a[idx], np.arange(1, a.size + 1)

    # sApar, xA = _sorted_mag(Apar_kHz)
    # sAperp, xP = _sorted_mag(Aperp_kHz)
    sfplus, xF = _sorted_mag(fplus_kHz)
    sfminus, xM = _sorted_mag(fminus_kHz)

    import matplotlib.ticker as mticker

    def _sorted_mag(vals):
        vals = np.asarray(vals, float)
        vals = np.abs(vals)  # sort by magnitude
        idx = np.argsort(vals)
        return vals[idx], np.arange(1, vals.size + 1)

    def _dual_log_plot(x1, y1, x2, y2, label1, label2, ylabel, title, annotate=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        # 1) filter to strictly positive y for log scale
        m1 = np.asarray(y1, float) > 0
        m2 = np.asarray(y2, float) > 0
        ax.plot(np.asarray(x1)[m1], np.asarray(y1)[m1], ".", ms=2, label=label1)
        ax.plot(np.asarray(x2)[m2], np.asarray(y2)[m2], ".", ms=2, label=label2)

        # 2) set log scale FIRST, with explicit base
        ax.set_yscale("log", base=10)

        # 3) robust limits spanning at least a decade (helps minors show)
        y_all_pos = np.concatenate([np.asarray(y1)[m1], np.asarray(y2)[m2]])
        if y_all_pos.size:
            ymin = np.min(y_all_pos)
            ymax = np.max(y_all_pos)
            if ymin <= 0 or not np.isfinite(ymin):  # just in case
                ymin = 1e-3
            # pad to ensure >= ~1 decade span
            if ymax / ymin < 10:
                pad = 10 / (ymax / ymin)
                ymin /= np.sqrt(pad)
                ymax *= np.sqrt(pad)
            ax.set_ylim(ymin, ymax)

        # 4) explicit major/minor locators (avoid 'auto')
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())  # show ticks, not labels
        ax.minorticks_on()

        # 5) grids
        ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", alpha=0.20, linewidth=0.6)
        ax.grid(True, which="both", axis="x", alpha=0.15, linewidth=0.6)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Site index (sorted by magnitude)")
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.8)
        if annotate:
            ax.text(
                0.98,
                0.98,
                annotate,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=6),
            )


    # --- build sorted series (kHz everywhere) ---
    # sApar, xA = _sorted_mag(Apar_kHz)
    # sAperp, xP = _sorted_mag(Aperp_kHz)
    sfplus, xF = _sorted_mag(fplus_kHz)
    sfminus, xM = _sorted_mag(fminus_kHz)

    # --- formulas for annotation (plain LaTeX, no \displaystyle) ---
    fpm_formula = (
        r"$f_{\pm}=\omega_{-1}\pm f_I,\quad "
        r"\omega_{-1}=\sqrt{(f_I-A_{\parallel})^{2}+A_{\perp}^{2}}$"
    )

    # --- 1) |A_parallel| vs |A_perp| on same plot ---
    # _dual_log_plot(
    #     xA,
    #     sApar,
    #     xP,
    #     sAperp,
    #     label1=r"$|A_{\parallel}|$",
    #     label2=r"$|A_{\perp}|$",
    #     ylabel="Coupling (kHz)",
    #     title=rf"|A| sorted • NV {orientation} • {proj_txt} {title_suffix} (≤{distance_max} Å)",
    # )

    # --- 2) |f+| vs |f-| on same plot ---
    # _dual_log_plot(
    #     xF,
    #     sfplus,
    #     xM,
    #     sfminus,
    #     label1=r"$|f_{+}|$",
    #     label2=r"$|f_{-}|$",
    #     ylabel="Frequency (kHz)",
    #     title=rf"ESEEM lines sorted • NV {orientation} • {proj_txt}",
    #     annotate=fpm_formula,
    # )
    # after:
    # fplus_kHz, mF = _in_range(fplus_kHz)
    # fminus_kHz, mM = _in_range(fminus_kHz)
    # Convert to arrays once
    fminus_all_kHz = np.asarray(fminus_list, float)
    fplus_all_kHz  = np.asarray(fplus_list, float)

    # Your old single-branch sorted plots can still use _in_range on copies if you want.
    # But for the paired plot, pass the raw arrays and let the helper mask them jointly:
    plot_paired_essem_lines(
        f_minus_kHz=fminus_all_kHz,
        f_plus_kHz=fplus_all_kHz,
        freq_range_kHz=freq_range_khz,
        title=rf"ESEEM pairwise lines • NV {orientation}",
    )
    plt.show()


    # # --- small plotting helper with log grids + optional annotation ---
    # def _one_fig(x, y, ylabel, title, annotate=None):
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.plot(x, y, ".", ms=2)
    #     ax.set_yscale("log")

    #     # Major + minor grid on a log axis
    #     ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
    #     ax.grid(True, which="minor", axis="y", alpha=0.20, linewidth=0.6)
    #     ax.grid(True, which="both", axis="x", alpha=0.15, linewidth=0.6)
    #     ax.yaxis.set_minor_locator(mticker.LogLocator(subs="auto"))

    #     ax.set_ylabel(ylabel)
    #     ax.set_xlabel("Site index (sorted by magnitude)")
    #     ax.set_title(
    #         f"{title} • NV {orientation} • {proj_txt} {title_suffix} (≤{distance_max} Å)"
    #     )

    #     # Optional formula annotation (top-right)
    #     if annotate:
    #         ax.text(
    #             0.98,
    #             0.98,
    #             annotate,
    #             transform=ax.transAxes,
    #             ha="right",
    #             va="top",
    #             fontsize=11,
    #             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=6),
    #         )

    # # Use it as before; add formulas on f+ and f−
    # fpm_formula = (
    #     r"$f_{\pm}=\omega_{-1}\pm f_I,\quad "
    #     r"\omega_{-1}=\sqrt{(f_I-A_{\parallel})^{2}+A_{\perp}^{2}}$"
    # )
    # _one_fig(xA, sApar, r"$|A_{\parallel}|$ (kHz)", r"|A∥| sorted")
    # _one_fig(xP, sAperp, r"$|A_{\perp}|$ (kHz)", r"|A⊥| sorted")
    # _one_fig(xF, sfplus, r"$|f_{+}|$ (kHz)", r"|f+| sorted", annotate=fpm_formula)
    # _one_fig(xM, sfminus, r"$|f_{-}|$ (kHz)", r"|f-| sorted", annotate=fpm_formula)


def plot_paired_essem_lines(
    f_minus_kHz,
    f_plus_kHz,
    *,
    freq_range_kHz=(1.0, 10000.0),
    title="ESEEM pair lines (f₋ & f₊ per site)",
):
    """
    For each 13C site, plot f_- and f_+ as a vertical pair at one x-position.

    - x-axis: site index (sorted by mid-frequency)
    - y-axis: frequency (log-scale), with a vertical line joining f_- and f_+.
    """
    f_minus = np.asarray(f_minus_kHz, float)
    f_plus  = np.asarray(f_plus_kHz, float)

    if f_minus.shape != f_plus.shape:
        raise ValueError(
            "f_minus_kHz and f_plus_kHz must have the same shape "
            "(one (f-, f+) pair per site)."
        )

    lo, hi = map(float, freq_range_kHz)

    # joint mask: finite, positive, and within [lo, hi] in *both* branches
    m = (
        np.isfinite(f_minus) & np.isfinite(f_plus) &
        (f_minus > 0) & (f_plus > 0) &
        (f_minus >= lo) & (f_minus <= hi) &
        (f_plus  >= lo) & (f_plus  <= hi)
    )

    f_minus = f_minus[m]
    f_plus  = f_plus[m]

    if f_minus.size == 0:
        raise ValueError("No valid (f_-, f_+) pairs in requested range.")

    # sort by mid-frequency to make the plot more readable
    f_mid = 0.5 * (f_minus + f_plus)
    order = np.argsort(f_mid)

    f_minus = f_minus[order]
    f_plus  = f_plus[order]
    x = np.arange(1, f_minus.size + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # vertical sticks
    ax.vlines(x, f_minus, f_plus, alpha=0.4, linewidth=0.8)
    # markers on f_- and f_+
    ax.plot(x, f_minus, ".", ms=3, label=r"$f_{-}$")
    ax.plot(x, f_plus,  ".", ms=3, label=r"$f_{+}$")

    ax.set_yscale("log", base=10)
    ax.set_ylim(9, 10000)

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("Site index (sorted by mid-frequency)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.85)

    return fig, ax

    

if __name__ == "__main__":
    # results = main()
    kpl.init_kplotlib()
    # Example:
    # R = make_R_NV((1, 1, -1))  # pick the NV orientation you’re simulating
    # res = suggest_distance_cutoff(
    #     hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
    #     R_NV=R,
    #     B_vec_T=B_vec_T,
    #     target_fraction=0.99,  # capture 99% of predicted modulation
    #     marginal_kappa_min=1e-4,  # ignore ultra-weak tail
    #     Ak_min_kHz=0.0,
    #     Ak_max_kHz=6000,  # keep your A∥ window (optional)
    #     Ak_abs=True,
    #     distance_max=None,
    # )
    # print("Suggested cutoff (Å or nm per your table):", res["cutoff_distance"])

    # You can also inspect / plot the cumulative curve:
    # dfc = res["table"]
    # Columns: distance, Apar_kHz, kappa_max, cum_kappa, cum_frac

    # # 1) Verify B differs across orientations
    # for ax in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]:
    #     check_axis(ax)
    #     R_NV = make_R_NV(ax)
    #     B_vec_NV = R_NV @ B_vec_T
    #     B_hat_NV = B_vec_NV / np.linalg.norm(B_vec_NV)
    #     print(ax, B_hat_NV)  # should be distinct
    # axes = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]

    # for ax in axes:
    # check_axis(ax)
    # 2) Pick one site; confirm A_par changes across orientations

    # site = sites[0]
    # for ax in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]:
    #     R = make_R_NV(ax)
    #     A_nv = (R @ site["A_crystal"] @ R.T) if IN_CRYSTAL else site["A_nv"]
    #     B_nv = R @ B_vec_T
    #     A_par, A_perp = compute_hyperfine_components(A_nv, B_nv / np.linalg.norm(B_nv))
    #     print(ax, A_par, A_perp)
    # Rotate B into NV frame once

    # A_par, A_perp = compute_hyperfine_components(A_any, B_hat_NV)
    # Single orientation (matches how you sorted your experimental list)
    # plot_sorted_hyperfine_and_essem(
    #     "analysis/nv_hyperfine_coupling/nv-2.txt",
    #     orientation=(1, 1, 1),
    #     distance_max=22.0,
    #     # show_bins=True,
    #     # nbins=30,
    #     title_suffix="(cutoff 22 Å, proj on B)",
    # )

    plot_sorted_hyperfine_and_essem(
        "analysis/nv_hyperfine_coupling/nv-2.txt",
        orientation=(1, 1, -1),
        distance_max=22.0,
        title_suffix="",
        project="B",
        freq_range_khz = (10,10000)
    )

    # Loop all four orientations if you want separate figures:
    # for ax in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]:
    #     plot_sorted_hyperfine_and_essem(
    #         "analysis/nv_hyperfine_coupling/nv-2.txt",
    #         orientation=ax,
    #         distance_max=22.0,
    #         title_suffix="",
    #         project="B",
    #     )
    plt.show(block=True)
