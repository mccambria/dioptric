# import matplotlib.pyplot as plt
# import numpy as np

# # Define parameters
# dead_time = 12  # Dead time in ms
# readout_times = np.arange(12, 121, 6)  # Range of readout times from 12 to 120 ms

# # Calculate cycle times and efficiency
# cycle_times = readout_times + dead_time
# efficiencies = readout_times / cycle_times
# measurement_rates = 1000 / cycle_times  # Measurements per second

# # Base SNR for minimum readout time
# base_snr = 0.1  # Arbitrary base SNR for 12 ms
# # snr_values = base_snr * np.sqrt(readout_times / 12)  # SNR scales as sqrt(T_readout)
# relative_snr = base_snr * np.sqrt(readout_times / 12)  # Relative improvement in SNR


# def calculate_best_readout_time(
#     dead_time, readout_times, efficiencies, measurement_rates
# ):
#     # Set thresholds for efficiency and measurement rate
#     efficiency_threshold = 0.67  # Minimum 67% efficiency
#     measurement_rate_threshold = 15  # Minimum 10 measurements per second

#     # Find the optimal readout time based on thresholds
#     optimal_time = None
#     for rt, eff, rate in zip(readout_times, efficiencies, measurement_rates):
#         if eff >= efficiency_threshold and rate >= measurement_rate_threshold:
#             optimal_time = rt
#             break

#     return optimal_time


# def create_plot(optimal_time):
#     plt.figure(figsize=(10, 5))
#     # Combine all metrics
#     plt.subplot(1, 2, 1)
#     plt.plot(readout_times, efficiencies * 100, marker="o", label="Efficiency (%)")
#     plt.plot(readout_times, cycle_times, marker="s", label="Cycle Time (ms)")
#     plt.plot(
#         readout_times, measurement_rates, marker="^", label="Measurement Rate (Hz)"
#     )
#     plt.axvline(
#         optimal_time,
#         color="g",
#         linestyle="--",
#         label=f"Optimal Time = {optimal_time} ms",
#     )
#     plt.title("Combined Metrics")
#     plt.xlabel("Readout Time (ms)")
#     plt.ylabel("Metrics")
#     plt.grid()
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(readout_times, relative_snr, marker="^", label="SNR")
#     plt.title("SNR vs Readout Time")
#     plt.xlabel("Readout Time (ms)")
#     plt.ylabel("Relative SNR")
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # Determine the optimal readout time
# optimal_readout_time = calculate_best_readout_time(
#     dead_time, readout_times, efficiencies, measurement_rates
# )
# print(f"Optimal Readout Time: {optimal_readout_time} ms")

# # Generate plots with optimal time highlighted
# create_plot(optimal_readout_time)


# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# revival_period = int(51.5e3 / 2)
# min_tau = 200
# revival_width = 2e3

# # Linear method sampling
# taus_linear = []
# taus_linear.extend(np.linspace(min_tau, min_tau + revival_width, 6).tolist())
# taus_linear.extend(
#     np.linspace(min_tau + revival_width, revival_period - revival_width, 7)[
#         1:-1
#     ].tolist()
# )
# taus_linear.extend(
#     np.linspace(
#         revival_period - revival_width, revival_period + revival_width, 61
#     ).tolist()
# )
# taus_linear.extend(
#     np.linspace(revival_period + revival_width, 2 * revival_period - revival_width, 7)[
#         1:-1
#     ].tolist()
# )
# taus_linear.extend(
#     np.linspace(
#         2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
#     ).tolist()
# )
# taus_linear = sorted(set(round(tau / 4) * 4 for tau in taus_linear))

# # Logarithmic method sampling
# taus_logarithmic = []
# taus_logarithmic.extend(
#     np.logspace(
#         np.log10(min_tau), np.log10(revival_period - revival_width), 10
#     ).tolist()
# )
# taus_logarithmic.extend(
#     np.linspace(
#         revival_period - revival_width, revival_period + revival_width, 21
#     ).tolist()
# )
# taus_logarithmic.extend(
#     np.linspace(revival_period + revival_width, 2 * revival_period - revival_width, 7)[
#         1:-1
#     ].tolist()
# )
# taus_logarithmic.extend(
#     np.linspace(
#         2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
#     ).tolist()
# )
# taus_logarithmic = sorted(set(round(tau / 4) * 4 for tau in taus_logarithmic))

# # Plot sampling
# plt.figure(figsize=(12, 6))

# # Linear method plot
# plt.subplot(1, 2, 1)
# plt.scatter(range(len(taus_linear)), taus_linear, color="blue", label="Linear Sampling")
# plt.title("Linear Sampling", fontsize=14)
# plt.xlabel("Step Index", fontsize=12)
# plt.ylabel("Tau (ns)", fontsize=12)
# plt.grid(alpha=0.3)
# plt.legend()

# # Logarithmic method plot
# plt.subplot(1, 2, 2)
# plt.scatter(
#     range(len(taus_logarithmic)),
#     taus_logarithmic,
#     color="green",
#     label="Logarithmic Sampling",
# )
# plt.title("Logarithmic Sampling", fontsize=14)
# plt.xlabel("Step Index", fontsize=12)
# plt.ylabel("Tau (ns)", fontsize=12)
# plt.grid(alpha=0.3)
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Compare number of steps
# len_linear = len(taus_linear)
# len_logarithmic = len(taus_logarithmic)

# len_linear, len_logarithmic

import matplotlib.pyplot as plt
import numpy as np

# fmt: off
# scc_duration_list = [168, 160, 164, 124, 188, 132, 116, 124, 160, 160, 164, 120, 140, 144, 124, 136, 136, 88, 152, 140, 140, 116, 104, 120, 112, 164, 136, 112, 96, 112, 140, 144, 196, 192, 120, 140, 228, 140, 32, 140, 148, 108, 164, 152, 132, 140, 176, 132, 136, 120, 112, 108, 144, 116, 132, 36, 192, 84, 148, 112, 132, 152, 176, 176, 176, 112, 120, 140, 168, 140, 92, 132, 92, 124, 68, 32, 92, 148, 164, 104, 32, 148, 188, 32, 112, 148, 168, 64, 140, 140, 96, 124, 176, 108, 108, 216, 216, 116, 112, 132, 148, 132, 132, 140, 160, 132, 148, 192, 160, 116, 140, 120, 152, 140, 144, 124, 160]
# scc_duration_list = [168, 184, 220, 136, 140, 104, 104, 144, 240, 188, 160, 148, 116, 164, 124, 140, 132, 104, 304, 184, 144, 148, 116, 68, 132, 120, 112, 124, 116, 148, 212, 144, 132, 172, 116, 160, 304, 144, 60, 180, 100, 112, 172, 192, 144, 184, 292, 200, 96, 116, 156, 144, 144, 80, 160, 160, 168, 76, 176, 136, 172, 192, 264, 140, 104, 112, 140, 176, 208, 148, 116, 140, 80, 152, 140, 116, 96, 120, 112, 96, 48, 188, 48, 84, 96, 228, 172, 172, 124, 96, 128, 120, 196, 104, 88, 140, 80, 116, 112, 160, 120, 140, 112, 148, 108, 140, 152, 292, 124, 116, 140, 140, 160, 212, 140, 140, 196]
# scc_duration_list = [112, 100, 92, 84, 144, 100, 100, 80, 108, 116, 92, 96, 108, 100, 88, 112, 108, 76, 76, 100, 132, 84, 92, 68, 76, 116, 124, 80, 100, 84, 76, 108, 128, 192, 92, 84, 92, 84, 108, 96, 132, 104, 116, 92, 100, 84, 92, 72, 84, 100, 116, 72, 124, 96, 84, 72, 164, 100, 56, 76, 64, 116, 92, 144, 172, 96, 60, 84, 100, 116, 80, 112, 88, 80, 64, 116, 100, 120, 112, 112, 128, 96, 108, 100, 108, 84, 144, 84, 128, 92, 108, 116, 148, 120, 88, 168, 64, 124, 104, 116, 100, 124, 112, 124, 120, 100, 172, 116, 124, 84, 92, 116, 80, 96, 88, 80, 92]
# scc_duration_list = [112, 100, 112, 76, 160, 108, 100, 92, 96, 100, 84, 92, 120, 108, 72, 100, 108, 72, 72, 124, 116, 84, 80, 80, 84, 156, 140, 92, 116, 72, 80, 124, 124, 128, 112, 84, 84, 92, 104, 104, 164, 92, 100, 92, 124, 72, 96, 100, 128, 104, 104, 68, 124, 92, 124, 100, 132, 100, 84, 132, 80, 104, 80, 172, 172, 116, 92, 92, 112, 124, 80, 136, 96, 104, 60, 88, 128, 144, 116, 116, 180, 96, 84, 108, 84, 100, 124, 272, 152, 76, 100, 108, 128, 116, 92, 152, 124, 140, 108, 120, 132, 156, 108, 160, 124, 96, 180, 100, 144, 92, 124, 116, 92, 112, 124, 108, 108]
# scc_duration_list = [136, 116, 116, 84, 180, 104, 108, 96, 84, 108, 128, 72, 144, 116, 84, 100, 116, 64, 84, 124, 116, 88, 92, 84, 80, 180, 132, 92, 120, 108, 92, 124, 108, 164, 132, 144, 100, 100, 144, 128, 216, 96, 124, 100, 84, 60, 92, 104, 108, 104, 96, 128, 116, 124, 88, 100, 168, 88, 72, 100, 76, 172, 44, 136, 272, 116, 100, 172, 128, 160, 80, 112, 104, 128, 104, 132, 80, 136, 112, 100, 128, 144, 136, 116, 96, 100, 200, 140, 128, 72, 108, 152, 212, 100, 88, 160, 124, 124, 124, 176, 272, 168, 184, 272, 164, 228, 208, 172, 272, 272, 264, 228, 216, 136, 176, 272, 164]
# scc_duration_list = [136, 112, 124, 88, 164, 104, 216, 84, 92, 116, 136, 88, 92, 120, 108, 100, 124, 52, 92, 124, 124, 100, 104, 80, 68, 156, 160, 108, 124, 104, 100, 116, 136, 168, 116, 168, 116, 116, 84, 156, 156, 84, 116, 80, 92, 64, 84, 108, 124, 120, 108, 172, 124, 136, 84, 128, 136, 108, 76, 100, 80, 108, 68, 156, 272, 112, 84, 180, 156, 184, 84, 108, 72, 128, 120, 120, 80, 140, 132, 88, 116, 120, 144, 92, 88, 112, 164, 128, 128, 64, 112, 196, 164, 92, 104, 168, 108, 132, 128, 196, 184, 164, 148, 272, 116, 216, 212, 236, 272, 204, 248, 272, 116, 176, 128, 232, 272]
# scc_duration_list = [128, 124, 136, 84, 144, 112, 124, 100, 108, 116, 140, 84, 120, 112, 112, 100, 116, 68, 100, 124, 136, 128, 100, 88, 80, 160, 144, 112, 112, 108, 108, 136, 124, 168, 124, 172, 136, 116, 84, 200, 144, 108, 124, 92, 100, 64, 96, 116, 92, 112, 100, 188, 188, 124, 92, 136, 140, 108, 80, 92, 84, 92, 76, 164, 272, 144, 92, 272, 160, 172, 92, 108, 80, 140, 140, 108, 88, 160, 120, 108, 140, 140, 148, 100, 100, 108, 164, 272, 116, 64, 164, 136, 152, 100, 104, 180, 96, 140, 164, 144, 272, 172, 136, 272, 136, 244, 272, 272, 272, 172, 272, 228, 120, 196, 144, 272, 180]
# scc_duration_list = [116, 108, 108, 72, 152, 104, 236, 96, 76, 108, 116, 84, 100, 108, 84, 92, 116, 68, 80, 104, 124, 108, 92, 76, 64, 152, 124, 88, 108, 92, 80, 112, 120, 164, 108, 116, 84, 116, 80, 124, 164, 92, 116, 80, 96, 64, 84, 116, 88, 100, 84, 128, 128, 108, 84, 144, 136, 92, 64, 104, 80, 104, 80, 124, 272, 100, 76, 108, 128, 128, 76, 120, 56, 104, 108, 96, 92, 136, 124, 100, 100, 108, 100, 84, 88, 92, 200, 116, 120, 72, 116, 116, 180, 112, 96, 136, 92, 108, 96, 196, 216, 136, 124, 260, 112, 164, 272, 140, 272, 128, 272, 272, 132, 192, 172, 188, 272]
# scc_duration_list = [128, 108, 100, 80, 160, 100, 92, 88, 84, 116, 112, 92, 104, 100, 96, 104, 100, 80, 84, 100, 128, 92, 84, 72, 64, 164, 136, 92, 124, 92, 96, 124, 116, 148, 112, 112, 92, 116, 80, 116, 172, 80, 124, 72, 84, 64, 116, 100, 72, 100, 92, 128, 100, 96, 84, 124, 136, 100, 92, 100, 84, 16, 92, 124, 272, 96, 84, 124, 156, 128, 72, 124, 64, 116, 120, 136, 92, 160, 108, 80, 84, 108, 92, 92, 100, 136, 160, 124, 112, 56, 128, 128, 204, 108, 104, 152, 84, 108, 100, 144, 208, 144, 132, 272, 132, 272, 252, 124, 272, 128, 208, 208, 92, 144, 136, 160, 272]
# scc_duration_list = [108, 100, 92, 72, 176, 108, 108, 80, 64, 120, 116, 88, 120, 108, 92, 96, 108, 60, 72, 88, 124, 80, 84, 72, 56, 140, 120, 92, 108, 76, 80, 104, 124, 136, 100, 108, 84, 116, 64, 112, 164, 80, 108, 80, 72, 48, 80, 112, 100, 108, 84, 112, 92, 108, 100, 132, 160, 76, 88, 116, 80, 92, 92, 124, 272, 92, 120, 116, 144, 116, 64, 136, 72, 112, 100, 88, 80, 112, 108, 84, 92, 144, 120, 92, 72, 104, 188, 100, 116, 60, 108, 104, 196, 84, 108, 120, 100, 112, 92, 172, 188, 124, 128, 272, 112, 272, 272, 160, 272, 144, 240, 272, 132, 172, 272, 272, 204]
# scc_duration_list = [128, 104, 92, 84, 152, 124, 128, 80, 100, 116, 108, 88, 120, 100, 92, 100, 112, 60, 76, 92, 164, 68, 84, 84, 64, 136, 136, 76, 92, 72, 76, 116, 144, 180, 96, 92, 96, 124, 80, 100, 164, 80, 108, 80, 92, 80, 84, 96, 80, 100, 92, 64, 116, 100, 84, 76, 188, 92, 72, 72, 72, 72, 72, 184, 140, 80, 68, 116, 160, 112, 72, 132, 84, 108, 48, 108, 96, 124, 112, 84, 96, 84, 84, 84, 84, 80, 124, 272, 124, 72, 100, 100, 160, 96, 72, 204, 72, 128, 84, 120, 116, 108, 128, 136, 108, 104, 148, 128, 144, 96, 100, 108, 72, 100, 80, 88, 80]
scc_duration_list = [136, 108, 96, 92, 208, 124, 100, 92, 88, 112, 108, 92, 108, 92, 100, 100, 116, 116, 64, 100, 136, 68, 92, 72, 60, 124, 116, 72, 92, 64, 72, 120, 124, 232, 92, 96, 96, 116, 84, 96, 144, 80, 116, 84, 100, 80, 84, 72, 80, 108, 84, 72, 136, 108, 100, 100, 188, 92, 64, 84, 60, 100, 76, 184, 152, 92, 68, 108, 160, 108, 72, 132, 80, 112, 60, 76, 104, 116, 108, 96, 96, 92, 84, 92, 84, 64, 124, 100, 124, 80, 108, 96, 136, 80, 80, 188, 188, 128, 84, 116, 124, 100, 100, 124, 112, 84, 196, 108, 124, 100, 100, 104, 76, 104, 84, 84, 84]
# fmt: on
print(
    np.median(scc_duration_list), np.mean(scc_duration_list), np.std(scc_duration_list)
)


# import numpy as np
# from scipy.spatial.transform import Rotation

# # Given Calibration Coordinates
# calibration_coords_pixel = np.array(
#     [[242.728, 64.946], [7.856, 74.462], [86.077, 231.364]]
# )

# calibration_coords_green = np.array(
#     [[95.27, 99.052], [120.108, 102.667], [109.947, 118.554]]
# )

# # Original Mapping
# old_pixel = np.array([107.51, 120.42])
# old_green = np.array([108.979, 106.471])

# # New Mapping
# new_pixel = np.array([107.51, 120.42])
# new_green = np.array([108.389, 106.086])

# # Compute the translation shift
# shift = new_green - old_green  # Compute the difference

# # Apply the shift to the original calibration coordinates
# updated_calibration_coords_green = calibration_coords_green + shift

# # Print Updated Calibration Green Coordinates
# print("Updated Calibration Green Coordinates:")
# print(updated_calibration_coords_green)


import sys

import matplotlib.pyplot as plt
import numpy as np
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl

# Load the three image datasets
# data1 = dm.get_raw_data(file_id=1765534203820, load_npz=True)  # Shallow NVs (No notch filter)
# data2 = dm.get_raw_data(file_id=1765547015748, load_npz=True)  # Shallow NVs (With notch filter)
# data3 = dm.get_raw_data(file_id=1738555345860, load_npz=True)  # Deep NVs

# data1 = dm.get_raw_data(file_id=1765628065653, load_npz=True)  # Shallow NVs (No notch filter)
# data2 = dm.get_raw_data(file_id=1765648437030, load_npz=True)  # Shallow NVs (With notch filter)
# data3 = dm.get_raw_data(file_id=1758030513244, load_npz=True)  # Deep NVs ref image

# same power but NV number is changed
# data0 = dm.get_raw_data(file_id=1758030513244, load_npz=True)  # Deep NVs ref image
# data1 = dm.get_raw_data(file_id=1764798303740, load_npz=True)  # Shallow 4 NVs
# data2 = dm.get_raw_data(file_id=1764826100492, load_npz=True)  # Shallow 52NVs
# data3 = dm.get_raw_data(file_id=1764415655836, load_npz=True)  # Shallow 89NVs
# data4 = dm.get_raw_data(file_id=1764406045571, load_npz=True)  # Shallow 161NVs

data1 = dm.get_raw_data(file_id=1765590222808, load_npz=True)  # Shallow NVs Side A
data2 = dm.get_raw_data(
    file_id=1766226975894, load_npz=True
)  # Shallow NVs Side B (Etched)
data3 = dm.get_raw_data(
    file_id=1766297631657, load_npz=True
)  # Shallow NVs Side B (Non-Etched)
file_names = [
    dm.get_file_name(file_id)
    for file_id in [1765590222808, 1766226975894, 1766297631657]
]
print(file_names)
# Extract image arrays
img1 = np.array(data1["img_array"]["img_array"], dtype=np.float64)
img2 = np.array(data2["img_array"]["img_array"], dtype=np.float64)
img3 = np.array(data3["img_array"]["img_array"], dtype=np.float64)

k_gain = 1.0  # Example value in e-/ADU
em_gain = 10  # Example electron multiplying gain
baseline = 300  # Example baseline ADU

img1 = widefield.adus_to_photons(
    img1, k_gain=k_gain, em_gain=em_gain, baseline=baseline
)
img2 = widefield.adus_to_photons(
    img2, k_gain=k_gain, em_gain=em_gain, baseline=baseline
)
img3 = widefield.adus_to_photons(
    img3, k_gain=k_gain, em_gain=em_gain, baseline=baseline
)
# Plot final image
kpl.init_kplotlib()
# Plot Image 1
fig1, ax1 = plt.subplots()
kpl.imshow(ax1, img1, title="SideA_Laser_INTI_520)", cbar_label="Photons")
ax1.axis("off")
plt.show()

# Plot Image 2
fig2, ax2 = plt.subplots()
kpl.imshow(ax2, img2, title="SideB_Etched_Laser_INTI_520", cbar_label="Photons")
ax2.axis("off")
plt.show()

# Plot Image 3
fig3, ax3 = plt.subplots()
kpl.imshow(ax3, img3, title="SideB_Non-Etched_Laser_INTI_520", cbar_label="Photons")
ax3.axis("off")
plt.show()
kpl.show(block=True)
# img0 = np.array(data0["ref_img_array"]["ref_img_array"], dtype=np.float64)
# img1 = np.array(data1["ref_img_array"]["ref_img_array"], dtype=np.float64)
# img2 = np.array(data2["ref_img_array"]["ref_img_array"], dtype=np.float64)
# img3 = np.array(data3["ref_img_array"]["ref_img_array"], dtype=np.float64)
# img4 = np.array(data4["ref_img_array"]["ref_img_array"], dtype=np.float64)

# convert to photons


def remove_outliers(data):
    """Remove outliers using the IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


# filtered_datasets = [remove_outliers(img.flatten()) for img in [img1, img2, img3, img4]]
# img1, img2, img3, img4 = filtered_datasets
# Compute background levels (mean, median, std)
bg1_mean, bg1_median, bg1_std = np.mean(img1), np.median(img1), np.std(img1)
bg2_mean, bg2_median, bg2_std = np.mean(img2), np.median(img2), np.std(img2)
bg3_mean, bg3_median, bg3_std = np.mean(img3), np.median(img3), np.std(img3)
# bg4_mean, bg4_median, bg4_std = np.mean(img4), np.median(img4), np.std(img4)

# Plot histograms to compare background distributions
plt.figure(figsize=(8, 6))
# plt.hist(
#     img0.flatten(),
#     bins=100,
#     alpha=0.5,
#     label="Ref.(Deep 161NVs)",
#     color="gray",
# )
plt.hist(
    img1.flatten(),
    bins=100,
    alpha=0.5,
    label="SideA_Laser_INTI_520",
    color="blue",
)

plt.hist(
    img2.flatten(),
    bins=100,
    alpha=0.5,
    label="SideB_Etched_Laser_INTI_520",
    color="orange",
)
plt.hist(
    img3.flatten(),
    bins=100,
    alpha=0.5,
    label="SideB_Non-Etched_Laser_INTI_520",
    color="red",
)

# plt.hist(
#     img4.flatten(),
#     bins=100,
#     alpha=0.5,
#     label="Shallow 161 NVs (Readout Power:1917uW)",
#     color="purple",
# )

plt.xlabel("Photons", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title("Background Levels", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
# plt.yscale("log")
plt.tight_layout()
plt.show()
kpl.show(block=True)
sys.exit()

# Print summary statistics
print(f"Image 1 - Mean: {bg1_mean:.2f}, Median: {bg1_median:.2f}, Std: {bg1_std:.2f}")
print(f"Image 2 - Mean: {bg2_mean:.2f}, Median: {bg2_median:.2f}, Std: {bg2_std:.2f}")
print(f"Image 3 - Mean: {bg3_mean:.2f}, Median: {bg3_median:.2f}, Std: {bg3_std:.2f}")
# # Apply outlier removal
# filtered_datasets = [remove_outliers(img.flatten()) for img in [img1, img2, img3, img4]]
# datasets = [img1, img2, img3, img4]
labels = [
    "Shallow 4 NVs (readout power:1416uW)",
    "Shallow 52 NVs(readout power:1416uW)",
    "Shallow 89 NVs(readout power:1917uW)",
    "Shallow 161 NVs(readout power:1416uW)",
]
colors = ["blue", "orange", "red", "purple"]

for i in range(4):
    plt.figure()
    plt.hist(
        filtered_datasets[i].flatten(),
        bins=100,
        alpha=0.5,
        label=labels[i],
        color=colors[i],
    )

    plt.xlabel("Photons", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title(f"Background Level - {labels[i]}", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Number of NVs in each dataset
nv_counts = [4, 52, 89, 161]

# Median background levels
median_values = [bg1_median, bg2_median, bg3_median, bg4_median]

# Plot median background levels vs number of NVs
plt.figure(figsize=(6, 5))
plt.plot(nv_counts, median_values, marker="o", linestyle="-", color="black")

plt.xlabel("Number of NVs", fontsize=15)
plt.ylabel("Median Background Level", fontsize=15)
plt.title("Background Levels", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.show()


# Retrieve and print file names
file_names = [
    dm.get_file_name(file_id=1764798303740),  # Shallow 4 NVs
    dm.get_file_name(file_id=1764826100492),  # Shallow 52 NVs
    dm.get_file_name(file_id=1764415655836),  # Shallow 89 NVs
    dm.get_file_name(file_id=1764406045571),  # Shallow 161 NVs
]

# Print file names
for i, file_name in enumerate(file_names, start=1):
    print(f"File {i}: {file_name}")
