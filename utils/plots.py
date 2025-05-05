import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import sys
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
carrier_freq = 100e6  # 100 MHz
sample_rate = 2e9  # 2 GS/s
duration_ns = 60  # each step duration
T = int(sample_rate * duration_ns * 1e-9)

phase_list_deg = [0, 90, 0, 90, 90, 0, 90, 0]
amp = 1.0
t = np.arange(T * len(phase_list_deg)) / sample_rate

signal = np.zeros_like(t)

for i, phase in enumerate(phase_list_deg):
    phi = np.deg2rad(phase)
    signal[i * T : (i + 1) * T] = amp * np.cos(
        2 * np.pi * carrier_freq * t[i * T : (i + 1) * T] + phi
    )

plt.plot(t * 1e6, signal)
plt.title("CPDD Microwave Drive with Phase Steps")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.show()

sys.exit


def gaussian_phase_1d(x, x0, sigma, amplitude):
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def initial_phase_pattern_1d(length, points):
    x = np.arange(length)
    phase = np.zeros(length, dtype=np.complex128)
    for point in points:
        x0 = point
        phase += np.exp(1j * x * x0)
    return np.angle(phase)


def compute_intensity_1d(phase):
    return (
        np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(np.exp(1j * phase))))) ** 2
    )


# Parameters
length = 256
sigma = 200
amplitude = 30
points = [64, 128, 192]  # Example point locations

# Create 1D grid
x = np.arange(length)

# Initial phase pattern
initial_phase = initial_phase_pattern_1d(length, points)

# Gaussian phase
gaussian_phase_profile = gaussian_phase_1d(x, length // 2, sigma, amplitude)

# Modified phase pattern
modified_phase = initial_phase + gaussian_phase_profile

# Compute intensities
initial_intensity = compute_intensity_1d(initial_phase)
modified_intensity = compute_intensity_1d(modified_phase)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Initial Phase")
plt.plot(x, initial_phase, "k")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.title("Initial Intensity")
plt.plot(x, initial_intensity, "r")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.title("Gaussian Phase Profile")
plt.plot(x, gaussian_phase_profile, "b")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.title("Modified Intensity")
plt.plot(x, modified_intensity, "g")
plt.grid(True)

plt.tight_layout()
plt.tight_layout()

# Save figure with current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
plt.savefig(save_path)

plt.show()

import numpy as np
import matplotlib.pyplot as plt


def generate_spot_array_phase(shape, spots):
    y, x = np.indices(shape)  # Generate the meshgrid of coordinates
    phase = np.zeros(shape, dtype=np.complex128)
    for x0, y0 in spots:
        phase += np.exp(1j * (x * x0 + y * y0))
    return np.angle(phase)


def blaze_phase_profile(shape, alpha, beta):
    x = np.linspace(-np.pi, np.pi, shape[1])
    y = np.linspace(-np.pi, np.pi, shape[0])
    xx, yy = np.meshgrid(x, y)
    return alpha * xx + beta * yy


def compute_intensity(phase):
    return (
        np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * phase))))) ** 2
    )


# Parameters
shape = (256, 256)
spot_positions = [(64, 64), (128, 128), (192, 192)]  # Example spot positions
alpha_values = [0, 0.5, 1.0]  # Different phase gradients in x
beta_values = [0, 0.5, 1.0]  # Different phase gradients in y

# Initial phase pattern
initial_phase = generate_spot_array_phase(shape, spot_positions)

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):
        # Blaze phase
        blaze_phase = blaze_phase_profile(shape, alpha, beta)

        # Modified phase pattern
        modified_phase = initial_phase + blaze_phase

        # Compute modified intensity
        modified_intensity = compute_intensity(modified_phase)

        axs[i, j].imshow(modified_intensity, cmap="hot")
        axs[i, j].set_title(f"alpha={alpha}, beta={beta}")
        axs[i, j].axis("off")

plt.tight_layout()
plt.show()
