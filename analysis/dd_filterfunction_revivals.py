import numpy as np
import matplotlib.pyplot as plt


def omega_13c(B_gauss: float) -> float:
    B_T = float(B_gauss) * 1e-4
    gamma_hz_per_t = 10.705e6
    fL = gamma_hz_per_t * B_T
    return 2.0 * np.pi * fL


def ou_psd_one_sided(omega: np.ndarray, sigma_dw: float, tau_c: float) -> np.ndarray:
    omega = np.asarray(omega, float)
    return 4.0 * (sigma_dw**2) * tau_c / (1.0 + (omega * tau_c) ** 2)


def lorentz_norm(omega: np.ndarray, omega0: float, gamma: float) -> np.ndarray:
    """
    Normalized Lorentzian L(ω) with ∫_{-∞}^{∞} L(ω) dω = 1.
    L(ω) = (1/π) * (γ/2) / ((ω-ω0)^2 + (γ/2)^2)
    """
    omega = np.asarray(omega, float)
    g2 = 0.5 * gamma
    return (1.0 / np.pi) * (g2 / ((omega - omega0) ** 2 + g2**2))


def peak_psd_one_sided(omega: np.ndarray, omega0: float, sigma_peak: float, gamma: float) -> np.ndarray:
    """
    One-sided PSD peak centered at omega0, with an interpretable strength:
      sigma_peak is RMS amplitude of the narrowband component (rad/s).

    We set S_peak(ω) = 2 * sigma_peak^2 * L(ω) for ω>=0 (one-sided convention).
    """
    return 2.0 * (sigma_peak**2) * lorentz_norm(omega, omega0, gamma)


def pulse_times_cpmg(N_pulses: int, tau: float) -> tuple[np.ndarray, float]:
    N = int(N_pulses)
    tau = float(tau)
    if N < 1:
        raise ValueError("Use N_pulses>=1 (Echo=1, XY4~4, XY8~8).")
    tpi = (2.0 * np.arange(1, N + 1) - 1.0) * tau
    T = 2.0 * N * tau
    return tpi, T


def toggling_Y_of_omega(omega: np.ndarray, pulse_times: np.ndarray, T: float) -> np.ndarray:
    omega = np.asarray(omega, float)
    pulse_times = np.asarray(pulse_times, float)

    edges = np.concatenate(([0.0], pulse_times, [T]))
    y = 1.0 - 2.0 * (np.arange(len(edges) - 1) % 2)  # +1,-1,...

    w = omega.copy()
    w[w == 0] = 1e-30

    exp_iwt = np.exp(1j * w[:, None] * edges[None, :])
    num = exp_iwt[:, 1:] - exp_iwt[:, :-1]
    Y = np.sum(y[None, :] * num, axis=1) / (1j * w)
    return Y


def chi_from_psd(omega: np.ndarray, S_omega: np.ndarray, Y_omega: np.ndarray) -> float:
    integrand = S_omega * (np.abs(Y_omega) ** 2)
    return float((1.0 / (2.0 * np.pi)) * np.trapz(integrand, omega))


def omega_grid_with_focus(
    wmin: float, wmax: float, *, omega0: float | None = None, gamma: float = 0.0,
    n_log: int = 5000, n_focus: int = 4000, focus_width: float = 30.0
) -> np.ndarray:
    """
    Log grid for global coverage + linear dense grid around omega0 ± focus_width*gamma.
    """
    w_log = np.logspace(np.log10(wmin), np.log10(wmax), int(n_log))

    if omega0 is None or gamma <= 0:
        return w_log

    lo = max(wmin, omega0 - focus_width * gamma)
    hi = min(wmax, omega0 + focus_width * gamma)
    if hi <= lo:
        return w_log

    w_focus = np.linspace(lo, hi, int(n_focus))
    w = np.unique(np.concatenate([w_log, w_focus]))
    return w


def simulate_dd_with_revivals(
    N_pulses: int,
    tau_array: np.ndarray,
    *,
    B_gauss: float = 50.0,
    # OU background (often kills revivals if too strong)
    sigma_ou_hz: float = 2e3,
    tau_c: float = 300e-6,
    # revival peak
    sigma_peak_hz: float = 3e3,
    peak_width_hz: float = 500.0,
    # ω grid
    fmin_hz: float = 200.0,
    fmax_hz: float = 50e6,
    n_log: int = 5000,
    n_focus: int = 6000,
) -> dict:
    tau_array = np.asarray(tau_array, float)
    if np.any(tau_array <= 0):
        raise ValueError("tau_array must be > 0")

    omegaL = omega_13c(B_gauss)
    gammaL = 2.0 * np.pi * float(peak_width_hz)

    wmin = 2.0 * np.pi * float(fmin_hz)
    wmax = 2.0 * np.pi * float(fmax_hz)
    omega = omega_grid_with_focus(
        wmin, wmax, omega0=omegaL, gamma=gammaL,
        n_log=n_log, n_focus=n_focus, focus_width=40.0
    )

    sigma_ou = 2.0 * np.pi * float(sigma_ou_hz)
    sigma_pk = 2.0 * np.pi * float(sigma_peak_hz)

    W = np.empty_like(tau_array)
    W_peak_only = np.empty_like(tau_array)
    W_ou_only = np.empty_like(tau_array)
    chi_tot = np.empty_like(tau_array)
    chi_ou = np.empty_like(tau_array)
    chi_pk = np.empty_like(tau_array)
    T_arr = np.empty_like(tau_array)

    S_ou = ou_psd_one_sided(omega, sigma_dw=sigma_ou, tau_c=tau_c)
    S_pk = peak_psd_one_sided(omega, omega0=omegaL, sigma_peak=sigma_pk, gamma=gammaL)
    S_tot = S_ou + S_pk

    for i, tau in enumerate(tau_array):
        tpi, T = pulse_times_cpmg(N_pulses, tau)
        T_arr[i] = T
        Y = toggling_Y_of_omega(omega, tpi, T)

        chi_ou[i] = chi_from_psd(omega, S_ou, Y)
        chi_pk[i] = chi_from_psd(omega, S_pk, Y)
        chi_tot[i] = chi_ou[i] + chi_pk[i]

        W_ou_only[i] = np.exp(-chi_ou[i])
        W_peak_only[i] = np.exp(-chi_pk[i])
        W[i] = np.exp(-chi_tot[i])

    return dict(
        tau=tau_array, T=T_arr,
        W=W, W_peak_only=W_peak_only, W_ou_only=W_ou_only,
        chi=chi_tot, chi_ou=chi_ou, chi_pk=chi_pk,
        omegaL=omegaL
    )


def main():
    # If you want "classic" visible revivals, use ~30–80 G (period ~10–30 µs)
    B = 50.0
    tau = np.linspace(0.2e-6, 200e-6, 900)  # half-spacing
    res = simulate_dd_with_revivals(
        N_pulses=8,        # dd
        tau_array=tau,
        B_gauss=B,
        sigma_ou_hz=1500.0,    #smears
        tau_c=400e-6,
        sigma_peak_hz=15000.0,  # hihger -> stronger revivals
        peak_width_hz=1500.0,   # narrower -> sharper revivals
        fmax_hz=60e6,
        n_log=5000,
        n_focus=8000,
    )

    fL = res["omegaL"] / (2*np.pi)
    print(f"B={B:g} G  ->  fL={fL/1e3:.2f} kHz,  period={1/fL*1e6:.2f} µs")

    T_us = res["T"] * 1e6

    plt.figure(figsize=(8,4.2))
    plt.plot(T_us, res["W_peak_only"], label="peak only (should show revivals)")
    plt.plot(T_us, res["W_ou_only"],   label="OU only (smooth decay)")
    plt.plot(T_us, res["W"],           label="total")
    plt.xlabel("Total evolution time T (µs)")
    plt.ylabel("W(T)")
    plt.title("Echo coherence: isolate peak-vs-OU contributions")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
