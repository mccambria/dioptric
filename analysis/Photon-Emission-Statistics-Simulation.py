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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup
wavelengths = np.linspace(400, 700, 1000)
sigma_emi = 4
max_wavelength_emi = 600
num_peaks = 6
lambda_poisson = 5

# Prepare figure
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
(line,) = ax.plot([], [], lw=2, color="orange", label="Emission Spectrum")
fill = ax.fill_between(wavelengths, 0, 0, color="orange", alpha=0.3)
ax.set_xlim(400, 700)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Wavelength (nm)", fontsize=14)
ax.set_ylabel("Intensity (a.u.)", fontsize=14)
ax.set_title("Time-Dependent Emission Spectrum (Poisson Random Peaks)", fontsize=13)
ax.grid(True)
ax.legend(fontsize=12)


# Animation function
def update(frame):
    emission_spectrum = np.zeros_like(wavelengths)
    poisson_emi = np.random.poisson(lambda_poisson, num_peaks)
    for i, intensity in enumerate(poisson_emi):
        peak_position = max_wavelength_emi - i * 15
        emission_spectrum += intensity * np.exp(
            -0.5 * ((wavelengths - peak_position) / sigma_emi) ** 2
        )
    emission_spectrum /= emission_spectrum.max()  # Normalize
    line.set_data(wavelengths, emission_spectrum)
    for coll in reversed(ax.collections):
        coll.remove()
    ax.fill_between(wavelengths, emission_spectrum, color="orange", alpha=0.3)
    return (line,)


# Create animation
# ani = FuncAnimation(fig, update, frames=100, interval=600, blit=False)

# Save or show
# ani.save("emission_spectrum_movie.mp4", writer="ffmpeg", fps=5)  # Optional saving
# ani.save("emission_spectrum_movie.gif", writer="pillow", fps=2, dpi=200)

# Simulate Data (replace with your real data if available)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelengths = np.linspace(400, 700, 300)
sigma_emi = 4
max_wavelength_emi = 600
num_peaks = 6
lambda_poisson = 5
n_frames = 30  # Number of time steps

# Generate time-dependent spectra
spectra_over_time = []

for frame in range(n_frames):
    emission_spectrum = np.zeros_like(wavelengths)
    poisson_emi = np.random.poisson(lambda_poisson, num_peaks)
    for i, intensity in enumerate(poisson_emi):
        peak_position = max_wavelength_emi - i * 15
        emission_spectrum += intensity * np.exp(
            -0.5 * ((wavelengths - peak_position) / sigma_emi) ** 2
        )
    emission_spectrum /= emission_spectrum.max()
    spectra_over_time.append(emission_spectrum)

# Convert to 2D array (time × wavelength)
spectra_image = np.array(spectra_over_time)

# Plot 2D heatmap
plt.figure(figsize=(8, 5), dpi=200)
extent = [wavelengths.min(), wavelengths.max(), 0, n_frames]
plt.imshow(spectra_image, aspect="auto", extent=extent, origin="lower", cmap="inferno")

plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Time (frames)", fontsize=14)
plt.title("Time-Dependent Emission Spectrum", fontsize=13)
plt.colorbar(label="Normalized Intensity")
plt.tight_layout()
plt.show()
