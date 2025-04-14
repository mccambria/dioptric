# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations

# Magnetic Field vs. Distance
B_current = 41.6  # Gauss
B_target = 80.0  # Gauss
scaling_factor = (B_current / B_target) ** (1 / 3)
print(f"r₂ / r₁ = {scaling_factor:.3f}")


#######
def estimate_magnetic_field(splittings_MHz):
    gamma_e = 2.8  # MHz/G
    # Normalize tetrahedral NV orientation unit vectors
    nv_axes = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ]
    ) / np.sqrt(3)
    # Convert ESR splittings to B projections (in Gauss)
    b_parallel = np.array(splittings_MHz) / (2 * gamma_e)
    # Solve A·B = b in least-squares sense
    B_vec_full, residuals_full, *_ = np.linalg.lstsq(nv_axes, b_parallel, rcond=None)
    B_mag_full = np.linalg.norm(B_vec_full)
    # Full 4-NV fit
    print(
        f"Full 4-NV fit → B = {np.round(B_vec_full, 2)}, |B| = {B_mag_full:.2f} G , residual = {np.round(residuals_full, 2)}"
    )
    # Try all 3-NV combinations
    print("\nAll 3-NV combinations:")
    B_mags = []
    for indices in combinations(range(4), 3):
        A_3 = nv_axes[list(indices)]
        b_3 = b_parallel[list(indices)]
        B_vec_3 = np.linalg.solve(A_3, b_3)
        B_mag_3 = np.linalg.norm(B_vec_3)
        B_mags.append(B_mag_3)
        print(f" NVs {indices} → B = {np.round(B_vec_3, 2)}, |B| = {B_mag_3:.2f} G")
    print(f"average B magnitude from 3-NVs fit: {np.mean(B_mags):.2f} G")
    # residuals = []
    # for i in range(4):
    #     n_i = nv_axes[i]
    #     pred_split = 2 * gamma_e * np.dot(B_vec_full, n_i)
    #     actual_split = splittings[i]
    #     res = np.abs(pred_split - actual_split)
    #     residuals.append(res)
    #     print(f"Residual for NV {i}: {res:.2f} MHz")


# Your 4 ESR splittings in MHz
splittings = [186, 145, 68, 25]  # old spliting
# splittings = [214, 162, 90, 45]  # new spliting
estimate_magnetic_field(splittings)


def estimate_revival_time(B_gauss):
    gamma_nuc_on_2pi = 1.071  # KHz/G
    omega = gamma_nuc_on_2pi * B_gauss  # Larmor frequency in KHz
    omega = omega / 1e3  # Larmor frequency in MHz
    T_rev = 1 / (omega)  # Revival time in microseconds
    return T_rev, omega  # Return revival time in nanoseconds


# Example: 80 Gauss
rev_time_us, larmor_freq_MHz = estimate_revival_time(44)
# rev_time_us, larmor_freq_MHz = estimate_revival_time(84)

print(f"Larmor frequency: {larmor_freq_MHz:.2f} Khz")
print(f"Revival time: {rev_time_us:.2f} us")


# def get_collapse_and_revival_times(B_gauss, num_revivals=1):
#     """
#     Calculate collapse and revival times in spin echo / XY8 experiments due to
#     Larmor precession of 13C nuclear spins.

#     Parameters:
#     - B_gauss: Magnetic field in Gauss
#     - num_revivals: Number of revival times to compute

#     Returns:
#     - dict with Larmor frequency (kHz), collapse time (µs), and revival times (µs)
#     """
#     gamma_c13_kHz_per_G = 1.071  # kHz/G
#     omega_nuc_kHz = gamma_c13_kHz_per_G * B_gauss

#     t_collapse_us = np.pi / omega_nuc_kHz
#     t_revivals_us = [2 * n * np.pi / omega_nuc_kHz for n in range(1, num_revivals + 1)]

#     return {
#         "B_field_G": B_gauss,
#         "omega_nuc_kHz": omega_nuc_kHz,
#         "collapse_time_us": t_collapse_us,
#         "revival_times_us": t_revivals_us,
#     }


# def print_revival_table(B_fields, num_revivals=1):
#     print(
#         f"{'B (G)':>6} | {'ω_nuc (kHz)':>12} | {'Collapse Time (µs)':>20} | Revival Times (µs)"
#     )
#     print("-" * 70)
#     for B in B_fields:
#         info = get_collapse_and_revival_times(B, num_revivals)
#         revivals_str = ", ".join([f"{t:.2f}" for t in info["revival_times_us"]])
#         print(
#             f"{B:6.1f} | {info['omega_nuc_kHz']:12.2f} | {info['collapse_time_us']:20.2f} | {revivals_str}"
#         )


# # Example usage
# B_fields = [37, 50, 84, 150]  # Magnetic fields in Gauss
# print_revival_table(B_fields, num_revivals=1)


# Optional plot
# def plot_revival_times(B_fields):
#     T_revs = [2 * np.pi / (1.071 * B) for B in B_fields]  # µs
#     plt.plot(B_fields, T_revs, "o-", label="13C revival period")
#     plt.xlabel("Magnetic Field (G)")
#     plt.ylabel("Revival Period (µs)")
#     plt.title("13C Revival Period vs Magnetic Field")
#     plt.grid(True)
#     plt.legend()
#     plt.show()


# plot_revival_times(np.linspace(10, 100, 50))


# Spin-1 matrices
Sx = (1 / np.sqrt(2)) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

Sy = (1 / np.sqrt(2)) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])

Sz = np.diag([1, 0, -1])


def get_rotation_to_NV_frame(nv_axis):
    """
    Constructs a rotation matrix that aligns lab-frame Z with NV axis.
    """
    z_lab = np.array([0, 0, 1])
    v = np.cross(z_lab, nv_axis)
    s = np.linalg.norm(v)
    c = np.dot(z_lab, nv_axis)

    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
    return R


def hamiltonian(B_lab, nv_axis, D=2870, gamma=2.8):
    """
    Constructs the full spin-1 Hamiltonian for a given magnetic field in lab frame,
    with the Hamiltonian defined in the NV frame.
    """
    # Project B_lab into the NV frame
    R = get_rotation_to_NV_frame(nv_axis)  # rotation matrix
    B_nv = R @ B_lab  # rotate B into the NV frame

    Bx, By, Bz = B_nv
    H = D * Sz @ Sz + gamma * (Bx * Sx + By * Sy + Bz * Sz)
    return H


def get_esr_splittings(B_vec, nv_axes):
    splittings = []
    for nv in nv_axes:
        H = hamiltonian(B_vec, nv)
        eigvals = np.linalg.eigvalsh(H)
        # transitions = np.sort(np.abs(np.diff(np.sort(eigvals))))
        # splitting = transitions[-1]  # |+1> - |-1> gap
        # splittings.append(splitting)
        splitting = eigvals[2] - eigvals[0]  # E_{+1} - E_{-1}
        splittings.append(splitting)
    return np.array(splittings)


def objective(B_vec, nv_axes, splittings_exp):
    splittings_calc = get_esr_splittings(B_vec, nv_axes)
    print("calcualted splttings", splittings_calc)
    return np.sum((splittings_calc - splittings_exp) ** 2)


def estimate_field_exact_H(splittings_MHz, nv_axes):
    nv_axes = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ]
    ) / np.sqrt(3)

    # Initial guess: high field approx
    b_parallel = np.array(splittings_MHz) / (2 * 2.8)
    # B_init, _, _, _ = np.linalg.lstsq(nv_axes, b_parallel, rcond=None)

    # result = minimize(
    #     objective, B_init, args=(nv_axes, splittings_MHz), method="Nelder-Mead"
    # )
    from scipy.optimize import Bounds

    B_init = np.array([20, 30, 40])  # or from high-field estimate
    # bounds = Bounds([-100, -100, -100], [100, 100, 100])
    # result = minimize(
    #     objective,
    #     B_init,
    #     args=(nv_axes, splittings_MHz),
    #     bounds=bounds,
    #     method="L-BFGS-B",
    # )

    bounds = Bounds([-150, -150, -150], [150, 150, 150])  # in Gauss

    result = minimize(
        objective,
        B_init,
        args=(nv_axes, splittings_MHz),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )

    B_vec = result.x
    B_mag = np.linalg.norm(B_vec)
    return B_vec, B_mag, result.fun  # fun = final residual


# nv_axes = np.array(
#     [
#         [1, 1, 1],
#         [1, -1, -1],
#         [-1, 1, -1],
#         [-1, -1, 1],
#     ]
# ) / np.sqrt(3)

# splittings = [214, 162, 89, 55]  # MHz
# B_vec, B_mag, residual = estimate_field_exact_H(splittings, nv_axes)
# print("Estimated B vector (G):", np.round(B_vec, 2))
# print("B magnitude (G):", round(B_mag, 2))
# print("Residual:", residual)

# simulated = get_esr_splittings(B_vec, nv_axes)
# print("Measured splittings (MHz):", splittings)
# print("Simulated splittings (MHz):", np.round(simulated, 2))

# for nv_axis in nv_axes:
#     H = hamiltonian(B_vec, nv_axis)
#     eigvals = np.sort(np.linalg.eigvalsh(H))
#     print(f"Eigenvalues (MHz): {np.round(eigvals, 2)}")
