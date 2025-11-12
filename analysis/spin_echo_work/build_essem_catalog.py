import json, csv, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.sc_c13_hyperfine_sim_data_driven import read_hyperfine_table_safe, B_vec_T
# ---------- helpers you already have (shown for completeness) ----------
# Pauli/2:
Sx = 0.5 * np.array([[0, 1],[1, 0]], float)
Sy = 0.5 * np.array([[0,-1j],[1j,0]], complex)
Sz = 0.5 * np.array([[1, 0],[0,-1]], float)

def _build_U_from_orientation(orientation, phi_deg=0.0):
    ez = np.asarray(orientation, float); ez /= np.linalg.norm(ez)
    trial = np.array([1.0,-1.0,0.0])
    if abs(np.dot(trial/np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0,1.0,-1.0])
    ex = trial - np.dot(trial, ez)*ez; ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    U0 = np.column_stack([ex, ey, ez])
    phi = np.deg2rad(phi_deg)
    Rz = np.array([[np.cos(phi),-np.sin(phi),0.0],
                   [np.sin(phi), np.cos(phi),0.0],
                   [0.0,         0.0,        1.0]])
    return U0 @ Rz, ez

def essem_lines_by_diag(A_file_Hz, orientation=(1,1,1), B_lab_vec=None,
                        gamma_n_Hz_per_T=10.705e6, ms=-1, phi_deg=0.0):
    U, z_nv_cubic = _build_U_from_orientation(orientation, phi_deg=phi_deg)
    A_cubic = U @ A_file_Hz @ U.T

    B_lab = np.asarray(B_lab_vec, float)
    Bmag  = float(np.linalg.norm(B_lab))
    if Bmag == 0.0: raise ValueError("B field magnitude is zero.")
    bx, by, bz = (B_lab / Bmag)
    fI_Hz = gamma_n_Hz_per_T * Bmag
    HZ  = fI_Hz * (bx*Sx + by*Sy + bz*Sz)

    Aeff_vec = A_cubic @ z_nv_cubic
    Hhf = float(ms) * (Aeff_vec[0]*Sx + Aeff_vec[1]*Sy + Aeff_vec[2]*Sz)

    evals0  = np.linalg.eigvalsh(HZ)
    evalsms = np.linalg.eigvalsh(HZ + Hhf)

    fI_split      = float(abs(evals0[1]  - evals0[0]))
    omega_ms_split= float(abs(evalsms[1] - evalsms[0]))

    f_minus = abs(omega_ms_split - fI_split)
    f_plus  =      omega_ms_split + fI_split
    return f_minus, f_plus, fI_split, omega_ms_split, A_cubic, z_nv_cubic

# ---------- 1) Build and save the full catalog ----------
def build_essem_catalog(hyperfine_path,
                        B_lab_vec,
                        orientations=((1,1,1),(1,1,-1),(1,-1,1),(-1,1,1)),
                        distance_max_A=22.0,
                        gamma_n_Hz_per_T=10.705e6,
                        ms=-1,
                        phi_deg=0.0,
                        out_json="essem_freq_catalog.json",
                        out_csv="essem_freq_catalog.csv"):
    """
    Reads the hyperfine table, computes (f-, f+) for all sites within distance_max_A
    and all NV orientations; saves JSON+CSV with handy fields for later fitting.
    """
    df = read_hyperfine_table_safe(hyperfine_path).copy()
    df = df[df["distance"] <= float(distance_max_A)].reset_index(drop=True)

    # precompute B-hat for amplitude proxy
    B_lab = np.asarray(B_lab_vec, float)
    Bmag  = float(np.linalg.norm(B_lab))
    B_hat = B_lab / Bmag

    records = []
    for ori in orientations:
        for i, row in df.iterrows():
            # A_file in Hz (NV-(111) frame)
            A_file_Hz = np.array([[row.Axx, row.Axy, row.Axz],
                                  [row.Axy, row.Ayy, row.Ayz],
                                  [row.Axz, row.Ayz, row.Azz]], float) * 1e6

            fm, fp, fI, wms, A_cubic, z_nv = essem_lines_by_diag(
                A_file_Hz=A_file_Hz,
                orientation=ori,
                B_lab_vec=B_lab,
                gamma_n_Hz_per_T=gamma_n_Hz_per_T,
                ms=ms,
                phi_deg=phi_deg,
            )

            # Optional amplitude weight (dimensionless): ~ sin^2(theta)*(A_perp/omega)^2
            # Decompose A_cubic relative to B_hat
            A_par  = float(B_hat @ A_cubic @ B_hat)
            A_perp = float(np.linalg.norm(A_cubic @ B_hat - A_par * B_hat))
            cos_th = float(np.clip(B_hat @ (z_nv/np.linalg.norm(z_nv)), -1, 1))
            sin2_th= 1.0 - cos_th**2
            amp_wt = (A_perp / max(wms, 1e-30))**2 * sin2_th

            records.append({
                "orientation": tuple(int(x) for x in ori),
                "site_index": int(i),
                "distance_A": float(row["distance"]),
                "f_minus_Hz": float(fm),
                "f_plus_Hz":  float(fp),
                "fI_Hz":      float(fI),
                "omega_ms_Hz":float(wms),
                "A_par_Hz":   float(A_par),
                "A_perp_Hz":  float(A_perp),
                "amp_weight": float(amp_wt),
            })

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)

    # Save CSV
    keys = list(records[0].keys()) if records else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    return records  # also return in-memory list

# ---------- 2) Load & filter for fitting ----------
def load_candidates(path_json,
                    fmin_kHz=150.0, fmax_kHz=20000.0,
                    top_by_weight=None,
                    orientations=None):
    """
    Load catalog and select frequency pairs within [fmin, fmax] (kHz).
    Optionally restrict orientations and/or choose top-N by amp_weight.
    Returns a list of dicts ready to seed the fit.
    """
    with open(path_json, "r") as f:
        recs = json.load(f)

    def in_window(r):
        fm_k = r["f_minus_Hz"]/1e3
        fp_k = r["f_plus_Hz"]/1e3
        okm  = (fmin_kHz <= fm_k <= fmax_kHz)
        okp  = (fmin_kHz <= fp_k <= fmax_kHz)
        return okm and okp

    sel = [r for r in recs if in_window(r)]
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
        sel = [r for r in sel if tuple(r["orientation"]) in ori_set]

    if top_by_weight is not None and len(sel) > top_by_weight:
        sel = sorted(sel, key=lambda r: r["amp_weight"], reverse=True)[:top_by_weight]

    return sel


import matplotlib.pyplot as plt, matplotlib.ticker as mticker


# --- Load your precomputed catalog (from build_essem_catalog) ---
def load_catalog(path_json):
    with open(path_json, "r") as f:
        return json.load(f)

# --- Helpers: window + pick orientations ---
def select_records(recs, fmin_kHz=150.0, fmax_kHz=20000.0, orientations=None):
    out = []
    ori_set = {tuple(o) for o in orientations} if orientations else None
    for r in recs:
        fm_k = r["f_minus_Hz"]/1e3
        fp_k = r["f_plus_Hz"] /1e3
        if not (np.isfinite(fm_k) and np.isfinite(fp_k)): 
            continue
        if not (fmin_kHz <= fm_k <= fmax_kHz and fmin_kHz <= fp_k <= fmax_kHz):
            continue
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue
        out.append(r)
    return out

# --- Expected spectrum (no MC): weight = P(occupancy)*amplitude_weight ---
def expected_spectrum_kHz(recs, p_occ=0.011):
    fks = []
    wts = []
    for r in recs:
        w = p_occ * float(r.get("amp_weight", 1.0))
        fks.append(r["f_minus_Hz"]/1e3); wts.append(w)
        fks.append(r["f_plus_Hz"] /1e3); wts.append(w)
    return np.array(fks, float), np.array(wts, float)

# --- Monte Carlo realizations of Bernoulli occupancy ---
def mc_spectra_kHz(recs, p_occ=0.011, n_sims=200, nbins=250, fmin_kHz=150.0, fmax_kHz=20000.0):
    edges = np.logspace(np.log10(fmin_kHz), np.log10(fmax_kHz), nbins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    H = np.zeros((n_sims, nbins), float)

    # prepack arrays for speed
    f_all = np.array([r["f_minus_Hz"]/1e3 for r in recs] + [r["f_plus_Hz"]/1e3 for r in recs])
    w_all = np.array([r.get("amp_weight",1.0) for r in recs] + [r.get("amp_weight",1.0) for r in recs])

    # per-site Bernoulli on the *site*, applied to its two lines
    n = len(recs)
    for k in range(n_sims):
        occ = (np.random.random(n) < p_occ).astype(float)   # 0/1 for each site
        w_occ = np.repeat(occ, 2) * w_all                   # apply to both f- and f+ of same site
        hist, _ = np.histogram(f_all, bins=edges, weights=w_occ)
        H[k, :] = hist

    return centers, H

# --- Plot: expected sticks + MC band (percentiles) ---
def plot_probable_essem(catalog_json,
                        p_occ=0.011,
                        orientations=None,
                        f_range_kHz=(150.0, 20000.0),
                        nbins=300,
                        n_sims=300,
                        show_expected=True):
    recs = load_catalog(catalog_json)
    recs = select_records(recs, fmin_kHz=f_range_kHz[0], fmax_kHz=f_range_kHz[1], orientations=orientations)

    # Expected “stick” spectrum
    fk, wt = expected_spectrum_kHz(recs, p_occ=p_occ)

    # Monte Carlo envelope
    centers, H = mc_spectra_kHz(recs, p_occ=p_occ, n_sims=n_sims, nbins=nbins,
                                fmin_kHz=f_range_kHz[0], fmax_kHz=f_range_kHz[1])

    p16, p50, p84 = np.percentile(H, [16, 50, 84], axis=0)  # 1σ band

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")

    # MC band
    ax.fill_between(centers, p16, p84, alpha=0.25, step="mid", label="MC 1σ band")

    # Median
    ax.step(centers, p50, where="mid", lw=1.8, label="MC median")

    # Expected sticks (optional)
    if show_expected and fk.size:
        ax.vlines(fk, 0, wt, alpha=0.3, linewidth=1.0, label="Expected sticks (p·amp)")

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Relative intensity (arb.)")
    title = "Probable ESEEM spectrum"
    if orientations:
        title += f" • orientations={list(map(tuple, orientations))}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()

    
def plot_sorted_expected_sticks(recs, p_occ=0.011, f_range_kHz=(150, 20000)):
    # expected sticks
    fk, wt = expected_spectrum_kHz(recs, p_occ=p_occ)
    m = np.isfinite(fk) & (fk>=f_range_kHz[0]) & (fk<=f_range_kHz[1]) & (wt>0)
    fk = fk[m]; wt = wt[m]

    # sort by frequency magnitude (ascending), like your old plots
    order = np.argsort(fk)
    fk_s = fk[order]
    idx   = np.arange(1, fk_s.size+1)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(idx, fk_s, ".", ms=2, label="Expected sticks")
    ax.set_yscale("log", base=10)
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("Rank (sorted by frequency)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Sorted expected ESEEM sticks")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()



# --- Utilities on your saved records -----------------------------------------
def lines_from_recs(recs, orientations=None, fmin_kHz=150.0, fmax_kHz=20000.0):
    """Return arrays: freqs_kHz (2 per site), amp_weight (same length), and site_idx (one per pair)."""
    ori_set = {tuple(o) for o in orientations} if orientations else None
    freqs = []
    weights = []
    site_idx = []  # to keep pairs grouped (two entries per site)
    for i, r in enumerate(recs):
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue
        fm = r["f_minus_Hz"]/1e3
        fp = r["f_plus_Hz"]/1e3
        if not (np.isfinite(fm) and np.isfinite(fp)): 
            continue
        if (fmin_kHz <= fm <= fmax_kHz) and (fmin_kHz <= fp <= fmax_kHz):
            w = float(r.get("amp_weight", 1.0))
            freqs += [fm, fp]
            weights += [w, w]
            site_idx += [i, i]
    return np.array(freqs, float), np.array(weights, float), np.array(site_idx, int)



def expected_stick_spectrum_from_recs(recs, p_occ=0.011, orientations=None,
                                      f_range_kHz=(150, 20000), use_weights=True,
                                      merge_tol_kHz=2.0,  # group lines within this tol
                                      normalize=False,
                                      overlay_convolved=None):
    """
    Discrete 'expected' spectrum:
      intensity at each line = p_occ * amp_weight (if present).
    Lines closer than merge_tol_kHz are merged to avoid double-plotting.
    """
    # pull lines
    freqs, w, _ = lines_from_recs(recs, orientations, *f_range_kHz)
    if freqs.size == 0:
        raise ValueError("No lines in range.")
    amps = p_occ * (w if use_weights else np.ones_like(w))

    # sort by frequency
    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    # merge close-by lines (within merge_tol_kHz)
    f_merged = []
    a_merged = []
    if f.size:
        acc_f = f[0]
        acc_a = a[0]
        for f0, a0 in zip(f[1:], a[1:]):
            if abs(f0 - acc_f) <= merge_tol_kHz:
                # merge into current bin (weighted average for freq, sum for amp)
                new_a = acc_a + a0
                acc_f = (acc_f*acc_a + f0*a0) / (new_a + 1e-30)
                acc_a = new_a
            else:
                f_merged.append(acc_f)
                a_merged.append(acc_a)
                acc_f, acc_a = f0, a0
        f_merged.append(acc_f)
        a_merged.append(acc_a)

    f_stick = np.asarray(f_merged, float)
    a_stick = np.asarray(a_merged, float)

    if normalize and a_stick.sum() > 0:
        a_stick = a_stick / a_stick.sum()

    # plot sticks
    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_xscale("log")
    ax.vlines(f_stick, 0.0, a_stick, linewidth=1.0, alpha=0.9, label="Expected sticks")

    # optional overlay: your continuous curve function returning (x,y)
    if overlay_convolved is not None:
        xf, yf = overlay_convolved()
        ax.plot(xf, yf, lw=1.6, alpha=0.7, label="Convolved (overlay)")

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Expected intensity (arb.)" + (" (normalized)" if normalize else ""))
    title = f"Discrete expected spectrum (p_occ={p_occ:.3f})"
    if orientations:
        title += f" • orientations={list(map(tuple, orientations))}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # # 1) Build once (all orientations, ≤22 Å)
    # catalog = build_essem_catalog(
    #     hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
    #     B_lab_vec=B_vec_T,                 # your lab B (consistent units with gamma)
    #     gamma_n_Hz_per_T=10.705e6,        # if B in Tesla; use 10705.0 if B in Gauss
    #     distance_max_A=15.0,
    #     ms=-1,
    #     phi_deg=0.0,                       # set if your table needs an in-plane twist
    #     out_json="essem_freq_catalog_22A.json",
    #     out_csv="essem_freq_catalog_22A.csv",
    # )

    # # 2) Later, for fitting — pick only the band you want, e.g. 150 kHz–20 MHz,
    # #    and (optionally) keep the strongest lines by weight.
    # cands = load_candidates(
    #     "essem_freq_catalog_22A.json",
    #     fmin_kHz=150.0,
    #     fmax_kHz=20000.0,
    #     top_by_weight=48,                 # e.g., keep top 48 lines
    #     orientations=[(1,1,1), (1,1,-1)] # or restrict to the NV families in your data
    # )
    
    
    recs = select_records(load_catalog("analysis/spin_echo_work/essem_freq_catalog_22A.json"),
                      fmin_kHz=150, fmax_kHz=20000, orientations=None)

    plot_sorted_expected_sticks(recs, p_occ=0.011, f_range_kHz=(10, 20000))

    expected_stick_spectrum_from_recs(
        recs, p_occ=0.011, orientations=None,
        f_range_kHz=(150, 20000), use_weights=True,
        merge_tol_kHz=2.0, normalize=False
    )
