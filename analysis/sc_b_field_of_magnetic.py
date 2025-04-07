# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""


import numpy as np
import matplotlib.pyplot as plt


splittings = [68, 186]  # in MHz
gamma_e = 2.8  # MHz/Gauss

B_parallel_1 = 68 / (2 * gamma_e)  # = 68 / 5.6
B_parallel_2 = 185 / (2 * gamma_e)  # = 186 / 5.6

print(f"B_parallel 1: {B_parallel_1:.2f} G")
print(f"B_parallel 2: {B_parallel_2:.2f} G")


B1 = 68 / 5.6  # = 12.14 G
B2 = 186 / 5.6  # = 33.21 G
cos_theta = -1 / 3

B_est = np.sqrt(B1**2 + B2**2 - 2 * B1 * B2 * cos_theta)
print(f"Estimated |B| ≈ {B_est:.2f} G")

# Magnetic Field vs. Distance
B_current = 41.6  # Gauss
B_target = 80.0  # Gauss

scaling_factor = (B_current / B_target) ** (1 / 3)
print(f"r₂ / r₁ = {scaling_factor:.3f}")


def estimate_magnetic_field_from_all_four(splittings_MHz):
    gamma_e = 2.8  # MHz/G

    # Normalize tetrahedral NV orientation unit vectors
    nv_axes = np.array(
        [
            [1, 1, 1],
            # [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ]
    ) / np.sqrt(3)

    # Convert ESR splittings to projected magnetic fields (in Gauss)
    b = np.array(splittings_MHz) / (2 * gamma_e)

    # Solve the linear system: A·B = b
    B_vec, _, _, _ = np.linalg.lstsq(nv_axes, b, rcond=None)
    B_mag = np.linalg.norm(B_vec)

    return B_vec, B_mag


# Your 4 ESR splittings in MHz
# splittings = [186, 68, 25, 145] #old spliting
splittings = [214, 162, 85, 50]  # new spliting
B_vec, B_mag = estimate_magnetic_field_from_all_four(splittings)

print("Estimated magnetic field vector (G):", np.round(B_vec, 2))
print("Estimated magnetic field magnitude (G):", round(B_mag, 2))
