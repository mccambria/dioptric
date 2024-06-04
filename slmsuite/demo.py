import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def gaussian_phase(x, y, x0, y0, sigma, amplitude):
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def initial_phase_pattern(shape, spots):
    phase = np.zeros(shape, dtype=np.complex128)  # Initialize as complex array
    for spot in spots:
        x0, y0 = spot
        phase += np.exp(1j * (x * x0 + y * y0)) #blaze phase
    return np.angle(phase)

def compute_intensity(phase):
    return np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.exp(1j * phase)))))**2

# Parameters
shape = (256, 256)
sigma = 200
amplitude = 15
spots = [(128, 128), (64, 64)]  # Example spot locations

# Create meshgrid
x = np.linspace(0, shape[0] - 1, shape[0])
y = np.linspace(0, shape[1] - 1, shape[1])
x, y = np.meshgrid(x, y)

# Initial phase pattern
initial_phase = initial_phase_pattern(shape, spots)

# Gaussian phase
gaussian_phase_profile = gaussian_phase(x, y, shape[0]//2, shape[1]//2, sigma, amplitude)

# Modified phase pattern
modified_phase = initial_phase + gaussian_phase_profile

# Compute intensities
initial_intensity = compute_intensity(initial_phase)
modified_intensity = compute_intensity(modified_phase)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Initial Phase")
plt.imshow(initial_phase, cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Initial Intensity")
plt.imshow(initial_intensity, cmap='hot')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Gaussian Phase Profile")
plt.imshow(gaussian_phase_profile, cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Modified Intensity")
plt.imshow(modified_intensity, cmap='hot')
plt.colorbar()

plt.tight_layout()
# Save figure with current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
plt.savefig(save_path)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Define grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Constants
x0 = 5
y0 = 5

# Complex exponential function
z = np.exp(1j * (x * x0 + y * y0))

# Plot real part
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.real(z), extent=(-5, 5, -5, 5), cmap='viridis')
plt.colorbar(label='Real part')
plt.title('Real part of exp(1j * (x * x0 + y * y0))')

# Plot imaginary part
plt.subplot(1, 2, 2)
plt.imshow(np.imag(z), extent=(-5, 5, -5, 5), cmap='viridis')
plt.colorbar(label='Imaginary part')
plt.title('Imaginary part of exp(1j * (x * x0 + y * y0))')

plt.tight_layout()
# Save figure with current date and time
# current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
# plt.savefig(save_path)
plt.show()
