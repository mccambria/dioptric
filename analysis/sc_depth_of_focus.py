# -*- coding: utf-8 -*-
# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
wavelengths = [520e-9, 589e-9, 638e-9]  # Wavelengths in meters
NA = 1.45  # Numerical aperture
n_oil = 1.5  # Refractive index of immersion oil
nv_depth = 10e-9  # NV depth in meters
diamond_thickness = 50e-6  # Diamond thickness in meters

# Compute Depth of Focus (DOF)
DOF_520 = (2 * n_oil * wavelengths[0]) / (NA**2)
DOF_589 = (2 * n_oil * wavelengths[1]) / (NA**2)
DOF_638 = (2 * n_oil * wavelengths[2]) / (NA**2)

# Define focus depth range for visualization
z_range = np.linspace(-2e-6, 2e-6, 500)  # Depth range (-2µm to 2µm)
focus_520 = np.exp(-((z_range / DOF_520) ** 2))  # Gaussian-like depth profile
focus_589 = np.exp(-((z_range / DOF_589) ** 2))  # Gaussian-like depth profile
focus_638 = np.exp(-((z_range / DOF_638) ** 2))  # Gaussian-like depth profile

# Plot focus profiles
plt.figure(figsize=(8, 5))
plt.plot(
    z_range * 1e6,
    focus_520,
    label=f"520 nm Laser (DOF = {DOF_520*1e6:.2f} µm)",
    color="green",
)
plt.plot(
    z_range * 1e6,
    focus_589,
    label=f"589 nm Laser (DOF = {DOF_589*1e6:.2f} µm)",
    color="orange",
)
plt.plot(
    z_range * 1e6,
    focus_638,
    label=f"638 nm Laser (DOF = {DOF_638*1e6:.2f} µm)",
    color="red",
)
# Mark NV layer position
plt.axvline(nv_depth * 1e6, color="red", linestyle="--", label="NV Depth (10 nm)")
plt.axvline(-nv_depth * 1e6, color="red", linestyle="--")

# Labels and customization
plt.xlabel("Depth (µm)", fontsize=14)
plt.ylabel("Relative Intensity (Focus Profile)", fontsize=14)
plt.title("Focus Depth Profile of Optical System", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# Define new depth range for imaging from the opposite side (50 µm thick diamond)
z_range_flipped = np.linspace(0, 60e-6, 1000)  # Depth range (-60 µm to 60 µm)

# Compute focus profiles when imaging through 50 µm of diamond
focus_520_flipped = np.exp(-(((z_range_flipped - 50e-6) / DOF_520) ** 2))
focus_589_flipped = np.exp(-(((z_range_flipped - 50e-6) / DOF_589) ** 2))
focus_638_flipped = np.exp(-(((z_range_flipped - 50e-6) / DOF_638) ** 2))

# Plot focus profiles
plt.figure(figsize=(8, 5))
plt.plot(
    z_range_flipped * 1e6,
    focus_520_flipped,
    label=f"520 nm Laser (DOF = {DOF_520*1e6:.2f} µm)",
    color="green",
)
plt.plot(
    z_range_flipped * 1e6,
    focus_589_flipped,
    label=f"589 nm Laser (DOF = {DOF_589*1e6:.2f} µm)",
    color="orange",
)
plt.plot(
    z_range_flipped * 1e6,
    focus_638_flipped,
    label=f"638 nm Laser (DOF = {DOF_638*1e6:.2f} µm)",
    color="red",
)

# Mark NV layer position at 50 µm depth
plt.axvline(50, color="red", linestyle="--", label="NV Depth (50 µm from surface)")

# Labels and customization
plt.xlabel("Depth (µm)", fontsize=14)
plt.ylabel("Relative Intensity (Focus Profile)", fontsize=14)
plt.title(
    "Focus Depth Profile When Imaging from Opposite Side (50 µm Thick Diamond)",
    fontsize=14,
)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
