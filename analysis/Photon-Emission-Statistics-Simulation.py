import numpy as np
import matplotlib.pyplot as plt
import sys

# Parameters
wavelengths = np.linspace(500, 700, 150)
num_peaks = 4
lambda_poisson = 3
sigma_emi = 4
max_wavelength_emi = 640
n_frames = 30  # Number of time steps

# Initialize storage
spectra_over_time = []

# Time loop
for frame in range(n_frames):
    emission_spectrum = np.zeros_like(wavelengths)

    # Generate Poisson-distributed intensities for this frame
    poisson_emi = np.random.poisson(lambda_poisson, num_peaks)

    # Add each peak
    for i, intensity in enumerate(poisson_emi):
        # Slight jitter: random shift within ±2 nm
        jitter = np.random.normal(loc=0, scale=2)
        peak_position = max_wavelength_emi - i * 15 + jitter
        # Inside the frame loop
        jitter = 4.0 * np.sin(2 * np.pi * frame / 40 + i)  # smooth oscillation
        jitter += np.random.normal(0, 0.2)  # small noise
        peak_position = max_wavelength_emi - i * 15 + jitter

        # Gaussian peak
        gaussian = intensity * np.exp(
            -0.5 * ((wavelengths - peak_position) / sigma_emi) ** 2
        )
        emission_spectrum += gaussian

    # Add noise
    noise = np.random.normal(0, 0.3, size=emission_spectrum.shape)
    emission_spectrum += noise
    emission_spectrum = np.clip(emission_spectrum, 0, None)

    # Normalize (optional, per-frame)
    emission_spectrum /= emission_spectrum.max() + 1e-9

    # Store
    spectra_over_time.append(emission_spectrum)

# Convert to 2D array
spectra_image = np.array(spectra_over_time)

# Plot 2D heatmap (time vs. wavelength)
plt.figure(figsize=(10, 5), dpi=200)
extent = [wavelengths.min(), wavelengths.max(), 0, n_frames]
plt.imshow(spectra_image, aspect="auto", extent=extent, origin="lower", cmap="inferno")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Time (frames)")
plt.title("Time-Dependent Emission Spectrum")
plt.colorbar(label="Normalized Intensity")
plt.tight_layout()
plt.show()


sys.exit()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters for spectra
wavelengths = np.linspace(300, 800, 1000)  # Wavelength range from 300 nm to 800 nm
absorption_spectrum = np.zeros_like(wavelengths)
emission_spectrum = np.zeros_like(wavelengths)

# Parameters for Gaussian broadening
sigma_abs = 5  # Standard deviation for absorption spectrum (in nm)
sigma_emi = 4  # Standard deviation for emission spectrum (in nm)
max_wavelength_abs = 400  # Peak wavelength for absorption
max_wavelength_emi = 600  # Peak wavelength for emission

# Poisson distribution parameters
lambda_poisson = 5  # Mean of the Poisson distribution
num_peaks = 6  # Number of peaks to simulate

# # Generate Poisson-distributed intensities for absorption and emission
poisson_abs = np.random.poisson(lambda_poisson, num_peaks)
poisson_emi = np.random.poisson(lambda_poisson, num_peaks)
# poisson_abs = poisson.pmf(np.arange(num_peaks), lambda_poisson) * num_peaks
# poisson_emi = poisson.pmf(np.arange(num_peaks), lambda_poisson) * num_peaks

# # Generate absorption spectrum
# for i, intensity in enumerate(poisson_abs):
#     peak_position = max_wavelength_abs + i * 15  # Shift each peak by 15 nm
#     absorption_spectrum += intensity * np.exp(-0.5 * ((wavelengths - peak_position) / sigma_abs)**2)

# Generate emission spectrum
for i, intensity in enumerate(poisson_emi):
    peak_position = max_wavelength_emi - i * 15  # Shift each peak by 15 nm
    emission_spectrum += intensity * np.exp(
        -0.5 * ((wavelengths - peak_position) / sigma_emi) ** 2
    )

# Normalize spectra
absorption_spectrum /= absorption_spectrum.max()
emission_spectrum /= emission_spectrum.max()

# Plot the spectra
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, absorption_spectrum, label="Absorption Spectrum", color="purple")
plt.plot(wavelengths, emission_spectrum, label="Emission Spectrum", color="orange")
plt.fill_between(wavelengths, absorption_spectrum, color="purple", alpha=0.3)
plt.fill_between(wavelengths, emission_spectrum, color="orange", alpha=0.3)

# Labeling
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Intensity (a.u.)", fontsize=14)
plt.title(
    "Absorption and Emission Spectra with Poisson-Distributed Intensities", fontsize=16
)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Setup
# wavelengths = np.linspace(400, 700, 1000)
# sigma_emi = 4
# max_wavelength_emi = 600
# num_peaks = 6
# lambda_poisson = 5

# # Prepare figure
# fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
# (line,) = ax.plot([], [], lw=2, color="orange", label="Emission Spectrum")
# fill = ax.fill_between(wavelengths, 0, 0, color="orange", alpha=0.3)
# ax.set_xlim(400, 700)
# ax.set_ylim(0, 1.1)
# ax.set_xlabel("Wavelength (nm)", fontsize=14)
# ax.set_ylabel("Intensity (a.u.)", fontsize=14)
# ax.set_title("Time-Dependent Emission Spectrum (Poisson Random Peaks)", fontsize=13)
# ax.grid(True)
# ax.legend(fontsize=12)


# # Animation function
# def update(frame):
#     emission_spectrum = np.zeros_like(wavelengths)
#     poisson_emi = np.random.poisson(lambda_poisson, num_peaks)
#     for i, intensity in enumerate(poisson_emi):
#         peak_position = max_wavelength_emi - i * 15
#         emission_spectrum += intensity * np.exp(
#             -0.5 * ((wavelengths - peak_position) / sigma_emi) ** 2
#         )
#     emission_spectrum /= emission_spectrum.max()  # Normalize
#     line.set_data(wavelengths, emission_spectrum)
#     for coll in reversed(ax.collections):
#         coll.remove()
#     ax.fill_between(wavelengths, emission_spectrum, color="orange", alpha=0.3)
#     return (line,)


# Create animation
# ani = FuncAnimation(fig, update, frames=100, interval=600, blit=False)

# Save or show
# ani.save("emission_spectrum_movie.mp4", writer="ffmpeg", fps=5)  # Optional saving
# ani.save("emission_spectrum_movie.gif", writer="pillow", fps=2, dpi=200)


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz  # For Voigt profile


# Voigt profile function
def voigt_profile(x, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


# Parameters
wavelengths = np.linspace(400, 700, 300)
num_peaks = 6
lambda_poisson = 5
n_frames = 30

# Thermal parameters
T0 = 5  # reference temperature in K
T_max = 7  # final temperature (simulate heating)

# Base peak properties
base_peak_positions = 600 - np.arange(num_peaks) * 15
sigma_G0 = np.random.uniform(1, 3, size=num_peaks)  # Gaussian width at T0
gamma_L0 = np.random.uniform(1, 3, size=num_peaks)  # Lorentzian width at T0
decay_constants = np.random.uniform(10, 25, size=num_peaks)

# Coefficients
alpha = 0.02  # linear increase in Lorentzian width with T
beta = 0.02  # redshift coefficient in nm/K

spectra_over_time = []

for frame in range(n_frames):
    emission_spectrum = np.zeros_like(wavelengths)

    # Temperature at this frame
    T = T0 + (T_max - T0) * frame / n_frames

    poisson_emi = np.random.poisson(lambda_poisson, num_peaks)

    for i in range(num_peaks):
        # Temperature-dependent linewidths
        sigma_G = sigma_G0[i] * np.sqrt(T / T0)
        gamma_L = gamma_L0[i] * (1 + alpha * (T - T0))

        # Temperature-induced redshift
        thermal_shift = beta * (T - T0)
        peak_position = base_peak_positions[i] + thermal_shift

        # Voigt peak
        peak = intensity * voigt_profile(wavelengths, peak_position, sigma_G, gamma_L)
        emission_spectrum += peak

    # Add noise
    noise = np.random.normal(0, 0.1, size=emission_spectrum.shape)
    emission_spectrum += noise

    # Normalize
    emission_spectrum = np.clip(emission_spectrum, 0, None)
    emission_spectrum /= emission_spectrum.max() + 1e-9

    spectra_over_time.append(emission_spectrum)

# Convert to 2D array
spectra_image = np.array(spectra_over_time)

# Plot
plt.figure(figsize=(8, 5), dpi=200)
extent = [wavelengths.min(), wavelengths.max(), 0, n_frames]
plt.imshow(spectra_image, aspect="auto", extent=extent, origin="lower", cmap="inferno")
plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Time (frames)", fontsize=14)
plt.title("Voigt-Shaped Spectra with Temperature Effects", fontsize=13)
plt.colorbar(label="Normalized Intensity")
plt.tight_layout()
plt.show()
