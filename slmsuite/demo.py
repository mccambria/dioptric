# from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np


# def gaussian_phase(x, y, x0, y0, sigma, amplitude):
#     return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


# def initial_phase_pattern(shape, spots):
#     phase = np.zeros(shape, dtype=np.complex128)  # Initialize as complex array
#     for spot in spots:
#         x0, y0 = spot
#         phase += np.exp(1j * (x * x0 + y * y0))  # blaze phase
#     return np.angle(phase)


# def compute_intensity(phase):
#     return (
#         np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.exp(1j * phase))))) ** 2
#     )


# # Parameters
# shape = (256, 256)
# sigma = 200
# amplitude = 15
# spots = [(128, 128), (64, 64)]  # Example spot locations

# # Create meshgrid
# x = np.linspace(0, shape[0] - 1, shape[0])
# y = np.linspace(0, shape[1] - 1, shape[1])
# x, y = np.meshgrid(x, y)

# # Initial phase pattern
# initial_phase = initial_phase_pattern(shape, spots)

# # Gaussian phase
# gaussian_phase_profile = gaussian_phase(
#     x, y, shape[0] // 2, shape[1] // 2, sigma, amplitude
# )

# # Modified phase pattern
# modified_phase = initial_phase + gaussian_phase_profile

# # Compute intensities
# initial_intensity = compute_intensity(initial_phase)
# modified_intensity = compute_intensity(modified_phase)

# # Plotting
# plt.figure(figsize=(12, 12))

# plt.subplot(2, 2, 1)
# plt.title("Initial Phase")
# plt.imshow(initial_phase, cmap="gray")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.title("Initial Intensity")
# plt.imshow(initial_intensity, cmap="hot")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.title("Gaussian Phase Profile")
# plt.imshow(gaussian_phase_profile, cmap="gray")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.title("Modified Intensity")
# plt.imshow(modified_intensity, cmap="hot")
# plt.colorbar()

# plt.tight_layout()
# # Save figure with current date and time
# current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
# plt.savefig(save_path)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Define grid
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)

# # Constants
# x0 = 5
# y0 = 5

# # Complex exponential function
# z = np.exp(1j * (x * x0 + y * y0))

# # Plot real part
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(np.real(z), extent=(-5, 5, -5, 5), cmap="viridis")
# plt.colorbar(label="Real part")
# plt.title("Real part of exp(1j * (x * x0 + y * y0))")

# # Plot imaginary part
# plt.subplot(1, 2, 2)
# plt.imshow(np.imag(z), extent=(-5, 5, -5, 5), cmap="viridis")
# plt.colorbar(label="Imaginary part")
# plt.title("Imaginary part of exp(1j * (x * x0 + y * y0))")

# plt.tight_layout()
# # Save figure with current date and time
# # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# # save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
# # plt.savefig(save_path)
# plt.show()


import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Given pixel coordinates and corresponding red coordinates
pixel_coords_list = np.array(
    [
        [95.439, 94.799],
        [119.117, 100.388],
        [109.423, 118.248],
    ]
)
red_coords_list = np.array(
    [
        [62.148, 62.848],
        [81.574, 67.148],
        [74.061, 81.78],
    ]
)

# For two points, a simpler method is necessary, but let's try using cv2.estimateAffinePartial2D
if len(pixel_coords_list) >= 3:
    # Use cv2.estimateAffinePartial2D to get the affine transformation matrix
    M = cv2.getAffineTransform(
        np.float32(pixel_coords_list), np.float32(red_coords_list)
    )

    # New pixel coordinate for which we want to find the corresponding red coordinate
    new_pixel_coord = np.array(
        [
            [107.748, 107.743],
            [118.127, 97.472],
            [107.036, 118.416],
            [96.822, 94.821],
        ],
        dtype=np.float32,
    )

    # Apply the affine transformation to the new pixel coordinate
    new_red_coord = cv2.transform(np.array([new_pixel_coord]), M)

    # Print the corresponding red coordinates
    print("[")
    for coord in new_red_coord[0]:
        rounded_coord = [round(x, 3) for x in coord]
        print(f"    {rounded_coord},")
    print("]")
else:
    # Calculate manually if only two points are available
    # Define the simple transformation function
    def simple_transform(pixel_point, src_points, dst_points):
        # Calculate scaling and translation manually
        scale_x = (dst_points[1][0] - dst_points[0][0]) / (
            src_points[1][0] - src_points[0][0]
        )
        scale_y = (dst_points[1][1] - dst_points[0][1]) / (
            src_points[1][1] - src_points[0][1]
        )

        # Calculate translation
        translation_x = dst_points[0][0] - scale_x * src_points[0][0]
        translation_y = dst_points[0][1] - scale_y * src_points[0][1]

        # Apply transformation
        new_x = scale_x * pixel_point[0] + translation_x
        new_y = scale_y * pixel_point[1] + translation_y

        return np.array([new_x, new_y])

    # New pixel coordinates to transform
    new_pixel_coord = np.array(
        [
            [120.137, 121.811],
            [133.937, 91.407],
            [76.778, 140.585],
            [160.878, 169.528],
        ],
        dtype=np.float32,
    )

    # Apply the transformation to each new pixel coordinate
    transformed_red_coords = [
        simple_transform(coord, pixel_coords_list, red_coords_list)
        for coord in new_pixel_coord
    ]

    # Print the transformed red coordinates
    print("[")
    for coord in transformed_red_coords:
        rounded_coord = [round(x, 3) for x in coord]
        print(f"    {rounded_coord},")
    print("]")

    # Calculate using simple linear transform
    # new_red_coord = simple_transform(
    #     [42.749, 125.763], pixel_coords_list, red_coords_list
    # )
    # print("Corresponding red coordinates:", new_red_coord)

min_tau = 200  # ns
max_tau = 100e3  # fallback if no revival_period given
taus = []

# Densely sample early decay
decay_width = 5e3
decay = np.linspace(min_tau, min_tau + decay_width, 6)
taus.extend(decay.tolist())

taus.extend(np.geomspace(min_tau + decay_width, max_tau, 81).tolist())

# Round to clock-cycle-compatible units
taus = [round(el / 4) * 4 for el in taus]

# Remove duplicates and sort
taus = sorted(set(taus))
taus_x = np.linspace(1, len(taus), len(taus))
plt.figure()
plt.scatter(taus_x, taus)
plt.show(block=True)
# sys.exit()


def generate_divisible_by_4(min_val, max_val, num_steps):
    step_size = (max_val - min_val) / (num_steps - 1)
    step_size = round(step_size / 4) * 4  # Ensure step size is divisible by 4

    values = [min_val + i * step_size for i in range(num_steps)]

    # Ensure values stay within bounds
    values = [x for x in values if x <= max_val]

    return values


# Example Usage
min_duration = 20
max_duration = 292
num_steps = 18

step_values = generate_divisible_by_4(min_duration, max_duration, num_steps)
print(step_values)
print(len(step_values))

sys.exit()
# Updating plot with center frequencies in the legend
# # Given data
# green_aod_freq_MHz = np.array([90, 95, 100, 105, 110, 115, 120, 125])
# green_laser_power_uW = np.array([260, 310, 330, 350, 360, 340, 240, 140])
# red_aod_freq_MHz = np.array([55, 60, 65, 70, 75, 80, 85, 90])
# red_laser_power_uW = np.array([112, 200, 255, 260, 270, 260, 205, 110])
# # Define center frequencies and compute x-axis difference
# green_center_freq = 110  # MHz
# red_center_freq = 75  # MHz
# green_x_diff = green_aod_freq_MHz - green_center_freq
# red_x_diff = red_aod_freq_MHz - red_center_freq
# # Normalize the laser powers using 0 uW as the minimum
# green_laser_power_normalized = green_laser_power_uW / green_laser_power_uW.max()
# red_laser_power_normalized = red_laser_power_uW / red_laser_power_uW.max()
# # Plotting
# plt.figure(figsize=(7, 5))
# plt.plot(
#     green_x_diff,
#     green_laser_power_normalized,
#     label="Green Laser Power (Center: 110 MHz)",
#     marker="o",
# )
# plt.plot(
#     red_x_diff,
#     red_laser_power_normalized,
#     label="Red Laser Power (Center: 75 MHz)",
#     marker="s",
# )
# plt.xlabel("Frequency Difference from Center (MHz)")
# plt.ylabel("Normalized Laser Power (uW)")
# plt.title("Normalized Laser Power vs Frequency Difference from Center")
# plt.legend()
# plt.grid(True)
# plt.show()
# Try a logarithmic function instead, which might better capture the relationship
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data points
aom_voltages = np.array(
    [0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45]
)
yellow_power = np.array([12, 25, 48, 86, 146, 236, 363, 540, 766, 1050, 1400])


# Define the exponential model
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


# Define models
def power_law_model(x, a, b, c):
    return a * x**b + c


# Fit the curve
params, covariance = curve_fit(power_law_model, aom_voltages, yellow_power)
# params, covariance = curve_fit(ower_law_model, aom_voltages, yellow_power)

# Extract parameters
a, b, c = params

# Generate fitted data for plotting
voltage_fit = np.linspace(aom_voltages.min(), aom_voltages.max(), 500)
# power_fit = exponential_model(voltage_fit, a, b, c)
power_fit = power_law_model(voltage_fit, a, b, c)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.scatter(aom_voltages, yellow_power, color="blue", label="Data Points")
plt.plot(
    voltage_fit,
    power_fit,
    color="red",
    label=f"Fit: $y = {a:.2f} \\cdot e^{{{b:.2f} \\cdot x}} + {c:.2f}$",
)
plt.title("AOM Voltage vs Yellow Laser Power")
plt.xlabel("AOM Voltage (V)")
plt.ylabel("Yellow Laser Power (µW)")
plt.legend()
plt.grid(True)
plt.show()

# Print the function
print(f"Fitted Function: y = {a:.3f} * exp({b:.3f} * x) + {c:.3f}")


# Define models


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data
aom_voltages = np.array(
    [0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45]
)
yellow_power = np.array([12, 25, 48, 86, 146, 236, 363, 540, 766, 1050, 1400])


# Goodness of fit (R^2 and RMSE)
def goodness_of_fit(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return r2, rmse


# Fit both models
models = {"Power-Law": power_law_model, "Exponential": exponential_model}

voltage_fit = np.linspace(aom_voltages.min(), aom_voltages.max(), 500)

plt.figure(figsize=(10, 6))
results = {}

for name, model in models.items():
    # Fit the model
    params, _ = curve_fit(
        model, aom_voltages, yellow_power, maxfev=10000, bounds=(0, np.inf)
    )
    power_fit = model(voltage_fit, *params)

    # Calculate goodness of fit
    y_fit = model(aom_voltages, *params)
    r2, rmse = goodness_of_fit(yellow_power, y_fit)
    results[name] = {"params": params, "R^2": r2, "RMSE": rmse}

    # Plot
    plt.plot(
        voltage_fit, power_fit, label=f"{name} Fit (R^2: {r2:.3f}, RMSE: {rmse:.2f})"
    )
    print(f"{name} Parameters: {params}, R^2: {r2:.3f}, RMSE: {rmse:.2f}")

# Plot data
plt.scatter(aom_voltages, yellow_power, color="black", label="Data Points")
plt.title("AOM Voltage vs Yellow Laser Power - Model Comparison")
plt.xlabel("AOM Voltage (V)")
plt.ylabel("Yellow Laser Power (µW)")
plt.legend()
plt.grid(True)
plt.show()

# Display results for better understanding
print("\nFitting Results:")
for model_name, res in results.items():
    print(f"{model_name}:")
    print(f"  Parameters: {res['params']}")
    print(f"  R^2: {res['R^2']:.3f}")
    print(f"  RMSE: {res['RMSE']:.2f}")


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import minimize

# # Data
# aom_voltages = np.array(
#     [0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45]
# )
# yellow_power = np.array([12, 25, 48, 86, 146, 236, 363, 540, 766, 1050, 1400])


# # Define models
# def power_law_model(x, a, b, c):
#     return a * x**b + c


# def exponential_model(x, a, b, c):
#     return a * np.exp(b * x) + c


# # Cost function for least squares
# def cost_function(params, x, y, model):
#     y_pred = model(x, *params)
#     residuals = y - y_pred
#     return np.sum(residuals**2)


# # Perform least squares fitting
# def fit_model_least_squares(x, y, model, initial_guess):
#     result = minimize(
#         cost_function,
#         initial_guess,
#         args=(x, y, model),
#         method="L-BFGS-B",
#         bounds=[(0, None)] * len(initial_guess),
#     )
#     return result.x, result.fun  # Optimized parameters and cost


# # Fit both models
# initial_guess = [1, 1, 1]  # Initial guesses for [a, b, c]
# power_params, power_cost = fit_model_least_squares(
#     aom_voltages, yellow_power, power_law_model, initial_guess
# )
# exp_params, exp_cost = fit_model_least_squares(
#     aom_voltages, yellow_power, exponential_model, initial_guess
# )

# # Generate fitted curves
# voltage_fit = np.linspace(aom_voltages.min(), aom_voltages.max(), 500)
# power_fit = power_law_model(voltage_fit, *power_params)
# exp_fit = exponential_model(voltage_fit, *exp_params)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(aom_voltages, yellow_power, color="black", label="Data Points")
# plt.plot(
#     voltage_fit,
#     power_fit,
#     label=f"Power-Law Fit (Cost: {power_cost:.2f})",
#     color="blue",
# )
# plt.plot(
#     voltage_fit, exp_fit, label=f"Exponential Fit (Cost: {exp_cost:.2f})", color="red"
# )
# plt.title("Least Squares Fit - AOM Voltage vs Yellow Laser Power")
# plt.xlabel("AOM Voltage (V)")
# plt.ylabel("Yellow Laser Power (µW)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Display results
# print("Fitting Results:")
# print(
#     f"Power-Law Model: a={power_params[0]:.3f}, b={power_params[1]:.3f}, c={power_params[2]:.3f}, Cost={power_cost:.2f}"
# )
# print(
#     f"Exponential Model: a={exp_params[0]:.3f}, b={exp_params[1]:.3f}, c={exp_params[2]:.3f}, Cost={exp_cost:.2f}"
# )
