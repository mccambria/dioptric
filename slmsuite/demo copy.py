import numpy as np
import matplotlib.pyplot as plt

def square_wave_1d(x, center, width, amplitude):
    """Generate a square wave centered at 'center' with given 'width' and 'amplitude'."""
    return amplitude * ((x >= center - width // 2) & (x <= center + width // 2))

def initial_phase_pattern_1d(length, points):
    """Generate the initial phase pattern."""
    x = np.arange(length)
    phase = np.zeros(length, dtype=np.complex128)
    for point in points:
        phase += np.exp(1j * x * point)
    return np.angle(phase)

def compute_intensity_1d(phase):
    """Compute intensity from phase pattern using Fourier Transform."""
    return np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(np.exp(1j * phase)))))**2

# Parameters
length = 256
amplitude = 30
points = [64, 128, 192]  # Example point locations
center = length // 2
width = 50  # Width of the square wave

# Create 1D grid
x = np.arange(length)

# Initial phase pattern
initial_phase = initial_phase_pattern_1d(length, points)

# Square wave phase profile
square_wave_profile = square_wave_1d(x, center, width, amplitude)

# Compute intensities for individual patterns
initial_intensity = compute_intensity_1d(initial_phase)
square_wave_intensity = compute_intensity_1d(square_wave_profile)

# Modified phase pattern by adding phases
modified_phase = initial_phase + square_wave_profile

# Compute intensity for the combined phase pattern
modified_intensity = compute_intensity_1d(modified_phase)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.title("Initial Phase")
plt.plot(x, initial_phase, 'k')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.title("Initial Intensity")
plt.plot(x, initial_intensity, 'r')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.title("Square Wave Phase Profile")
plt.plot(x, square_wave_profile, 'b')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.title("Square Wave Intensity")
plt.plot(x, square_wave_intensity, 'm')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.title("Combined Phase Pattern")
plt.plot(x, modified_phase, 'c')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.title("Combined Intensity Pattern")
plt.plot(x, modified_intensity, 'g')
plt.grid(True)

plt.tight_layout()
plt.show()
