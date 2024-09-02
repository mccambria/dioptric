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
import cv2
import numpy as np

# Given pixel coordinates and corresponding red coordinates
pixel_coords_list = np.array([[110.186, 129.281], [86.294, 103.0]])
red_coords_list = np.array([[75.49, 75.692], [77.801, 73.465]])

# For two points, a simpler method is necessary, but let's try using cv2.estimateAffinePartial2D
if len(pixel_coords_list) >= 3:
    # Use cv2.estimateAffinePartial2D to get the affine transformation matrix
    M, _ = cv2.estimateAffinePartial2D(
        np.float32(pixel_coords_list), np.float32(red_coords_list)
    )

    # New pixel coordinate for which we want to find the corresponding red coordinate
    new_pixel_coord = np.array([[128.233, 88.007]], dtype=np.float32)

    # Apply the affine transformation to the new pixel coordinate
    new_red_coord = cv2.transform(np.array([new_pixel_coord]), M)

    # Print the corresponding red coordinates
    print("Corresponding red coordinates:", new_red_coord[0][0])
else:
    # Calculate manually if only two points are available
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

    # Calculate using simple linear transform
    new_red_coord = simple_transform(
        [128.233, 88.007], pixel_coords_list, red_coords_list
    )
    print("Corresponding red coordinates:", new_red_coord)
