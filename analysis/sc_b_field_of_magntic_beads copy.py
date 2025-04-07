# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""


import numpy as np
import matplotlib.pyplot as plt

# Physical constants
mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Bead properties
beads = {
    "150 nm (SuperMag)": 75e-9,  # radius in meters
    "2.8 µm (MonoMag)": 1.4e-6,  # radius in meters
}

# Magnetic susceptibility (assumed - tweak based on data)
# chi = 0.5  # For superparamagnetic beads, rough estimate
chi = 0.1  # For superparamagnetic beads, rough estimate

# External magnetic fields in Gauss
B_ext_gauss = [10, 50, 100]
B_ext_tesla = [b * 1e-4 for b in B_ext_gauss]

# Distance range (in meters): 0–20 nm from bead surface
r_surface = np.linspace(1e-9, 50e-9, 200)

# Plotting
plt.figure(figsize=(10, 6))

for label, R in beads.items():
    V = (4 / 3) * np.pi * R**3  # volume of the bead

    for B_ext in B_ext_tesla:
        M = chi * B_ext / mu0  # Induced magnetization (A/m)
        m = M * V  # Dipole moment (A·m²)

        r = R + r_surface  # Absolute distance from center of bead
        Bz = (mu0 / (4 * np.pi)) * (2 * m) / (r**3)  # Bz field along dipole axis

        label_plot = f"{label}, B_ext = {B_ext*1e4:.0f} G"
        plt.plot(r_surface * 1e9, Bz * 1e4, label=label_plot)  # x in nm, B in mT

plt.xlabel("Distance from bead surface (nm)")
plt.ylabel("Magnetic Field Bz (G)")
plt.title("Estimated Magnetic Field of Paramagnetic Beads (Induced Dipole Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)

# Bead Parameters
R = 0.5e-6  # radius in meters
# R = 75e-9  # radius in meters

V = (4 / 3) * np.pi * R**3  # volume in m^3

# Saturation Magnetization (moderate value for Fe3O4 nanoparticle)
Ms = 3.5e5  # A/m

# Dipole moment
m = Ms * V  # A·m²

# Distances from bead surface (0 to 20 nm)
r_surface = np.linspace(1e-9, 20e-9, 200)
r = R + r_surface

# Magnetic field along dipole axis
Bz = (mu0 / (4 * np.pi)) * (2 * m) / (r**3)  # Tesla

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r_surface * 1e9, Bz * 1e3, label="Fe3O4 Core (1 µm Bead)")
plt.xlabel("Distance from bead surface (nm)")
plt.ylabel("Magnetic Field Bz (mT)")
plt.title("Magnetic Field near Amine Magnetic Bead (Fe₃O₄ Core, Dipole Approx.)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # H/m
chi = 0.1  # magnetic susceptibility (can adjust)
# chi = 100
# Bead radius and volume
R = 75e-9  # 150 nm bead
V = (4 / 3) * np.pi * R**3  # m³

# Distances (from bead surface)
r_surface = np.linspace(1e-9, 50e-9, 200)
r = R + r_surface  # total distance from center

# External magnetic fields in Tesla
B_ext_list = [10e-4, 50e-4, 100e-4]  # 10, 50, 100 Gauss

# Plot
plt.figure(figsize=(9, 5))

for B_ext in B_ext_list:
    M = chi * B_ext / mu0  # A/m
    m = M * V  # A·m²
    Bz = (mu0 / (4 * np.pi)) * (2 * m) / r**3  # Tesla
    plt.plot(r_surface * 1e9, Bz * 1e3, label=f"{B_ext*1e4:.0f} G")

plt.xlabel("Distance from bead surface (nm)")
plt.ylabel("Induced Bz field (mT)")
plt.title("Induced Magnetic Field near 150 nm Bead for External B Fields")
plt.legend(title="External Field")
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # H/m
R = 75e-9  # 150 nm radius
V = (4 / 3) * np.pi * R**3

# Distance from bead surface (1–20 nm)
r_surface = np.linspace(1e-9, 20e-9, 200)
r = R + r_surface

# Case 1: Induced (not saturated)
B_ext = 0.01  # Tesla = 100 G
chi = 80
M_induced = chi * B_ext / mu0
m_induced = M_induced * V
Bz_induced = (mu0 / (4 * np.pi)) * (2 * m_induced) / r**3

# Case 2: Saturated
Ms = 3.5e5  # A/m
m_sat = Ms * V
Bz_sat = (mu0 / (4 * np.pi)) * (2 * m_sat) / r**3

# Plotting
plt.figure(figsize=(9, 5))
plt.plot(
    r_surface * 1e9,
    Bz_induced * 1e3,
    label="Induced (χ = 80, B_ext = 100 G)",
    color="orange",
)
plt.plot(
    r_surface * 1e9,
    Bz_sat * 1e3,
    "--",
    label="Saturated (Ms = 3.5e5 A/m)",
    color="blue",
)
plt.xlabel("Distance from bead surface (nm)")
plt.ylabel("Bz (mT)")
plt.title("Magnetic Field near 150 nm Fe₃O₄ Bead")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


