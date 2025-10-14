import sys
import numpy as np
from matplotlib import pyplot as plt

from utils import kplotlib as kpl

def xy8_tau_grid(B_gauss=47.5,   # sets T_L
                 center_harmonics=(1,),  # (1,) or (1,3) to include 3rd
                 win_us=0.8,     # half-window for dense scan around each center
                 fine_step_us=0.1,
                 coarse_extra=False, coarse_margin_us=3.0, coarse_step_us=0.5,
                 clock_ns=4):    # snap to your OPX clock (4 ns typical)
    # 13C Larmor and period
    gamma_c13_MHz_per_T = 10.705
    TL_us = 1.0 / (gamma_c13_MHz_per_T * (B_gauss*1e-4)) * 1e6  # µs

    centers = []
    for k in center_harmonics:
        centers.append((2*k-1)*TL_us/2.0)  # τ centers at (2k-1) TL/2

    taus = []
    for c in centers:
        # dense window around each center
        dense = np.arange(c - win_us, c + win_us + 1e-9, fine_step_us)
        taus.extend(dense.tolist())

        # optional coarse flanks to see baseline shape
        if coarse_extra:
            left  = np.arange(max(0.2, c - coarse_margin_us - win_us), c - win_us,  coarse_step_us)
            right = np.arange(c + win_us, c + win_us + coarse_margin_us + 1e-9, coarse_step_us)
            taus.extend(left.tolist()); taus.extend(right.tolist())

    # snap to clock and clean up
    taus_ns = [round(tau_us*1e3/clock_ns)*clock_ns for tau_us in taus]  # to ns, snap
    taus_us = sorted(set([t/1e3 for t in taus_ns]))  # back to µs, unique & sorted
    return taus_us, TL_us

sys.exit()
def xy8_tau_grid(B_gauss=47.5,   # sets T_L
                 center_harmonics=(1,),  # (1,) or (1,3) to include 3rd
                 win_us=0.8,     # half-window for dense scan around each center
                 fine_step_us=0.1,
                 coarse_extra=False, coarse_margin_us=3.0, coarse_step_us=0.5,
                 clock_ns=4):    # snap to your OPX clock (4 ns typical)
    # 13C Larmor and period
    gamma_c13_MHz_per_T = 10.705
    TL_us = 1.0 / (gamma_c13_MHz_per_T * (B_gauss*1e-4)) * 1e6  # µs

    centers = []
    for k in center_harmonics:
        centers.append((2*k-1)*TL_us/2.0)  # τ centers at (2k-1) TL/2

    taus = []
    for c in centers:
        # dense window around each center
        dense = np.arange(c - win_us, c + win_us + 1e-9, fine_step_us)
        taus.extend(dense.tolist())

        # optional coarse flanks to see baseline shape
        if coarse_extra:
            left  = np.arange(max(0.2, c - coarse_margin_us - win_us), c - win_us,  coarse_step_us)
            right = np.arange(c + win_us, c + win_us + coarse_margin_us + 1e-9, coarse_step_us)
            taus.extend(left.tolist()); taus.extend(right.tolist())

    # snap to clock and clean up
    taus_ns = [round(tau_us*1e3/clock_ns)*clock_ns for tau_us in taus]  # to ns, snap
    taus_us = sorted(set([t/1e3 for t in taus_ns]))  # back to µs, unique & sorted
    return taus_us, TL_us

taus_us, TL_us = xy8_tau_grid(B_gauss=47.5, center_harmonics=(1,), win_us=0.8, fine_step_us=0.1)
plt.figure(figsize=(10,4))
plt.plot(taus_us, np.zeros_like(taus_us), 'o', label='First pass')
# plt.plot(taus_refine, np.ones_like(taus_refine)*0.2, 's', label='Refine')
# for k, c in centers.items():
#     plt.axvline(c, linestyle='--', color='r', alpha=0.6,
#                 label=f'Dip center (harmonic {k})')
plt.xlabel(r'$\tau$ (µs)')
plt.yticks([])
plt.legend(frameon=False)
# plt.title(f'Tau grids for XY8 around {list(centers.values())[0]:.2f} µs at 47.5 G')
plt.tight_layout()
plt.show()

print(f"T_L ≈ {TL_us:.2f} µs; first-dip center ≈ {TL_us/2:.2f} µs")
# -> feed 'taus_us' into your XY8 measurement loop (instead of spin echo taus)

import numpy as np
import matplotlib.pyplot as plt

# Reuse the function from earlier but inline for plotting taus
def xy8_tau_grids(B_gauss=47.5,
                  first_pass_halfwin_us=3.0, first_pass_step_us=0.2,
                  refine_halfwin_us=0.4, refine_step_us=0.05,
                  center_harmonics=(1,), clock_ns=4):
    gamma_c13_MHz_per_T = 10.705
    TL_us = 1.0 / (gamma_c13_MHz_per_T * (B_gauss * 1e-4)) * 1e6
    centers_us = {k: (2*k - 1) * TL_us / 2.0 for k in center_harmonics}
    def _snap_list(us_list):
        ns_list = [round(us * 1e3 / clock_ns) * clock_ns for us in us_list]
        return sorted(set([ns/1e3 for ns in ns_list]))
    taus_first, taus_refine = [], []
    for k, c in centers_us.items():
        taus_first.extend(np.arange(c-first_pass_halfwin_us,
                                    c+first_pass_halfwin_us+1e-12,
                                    first_pass_step_us).tolist())
        taus_refine.extend(np.arange(c-refine_halfwin_us,
                                     c+refine_halfwin_us+1e-12,
                                     refine_step_us).tolist())
    return centers_us, _snap_list(taus_first), _snap_list(taus_refine)

# Example for 47.5 G, first dip
centers, taus_first, taus_refine = xy8_tau_grids(B_gauss=47.5)

print(taus_first, taus_refine)
plt.figure(figsize=(10,4))
plt.plot(taus_first, np.zeros_like(taus_first), 'o', label='First pass')
plt.plot(taus_refine, np.ones_like(taus_refine)*0.2, 's', label='Refine')
for k, c in centers.items():
    plt.axvline(c, linestyle='--', color='r', alpha=0.6,
                label=f'Dip center (harmonic {k})')
plt.xlabel(r'$\tau$ (µs)')
plt.yticks([])
plt.legend(frameon=False)
plt.title(f'Tau grids for XY8 around {list(centers.values())[0]:.2f} µs at 47.5 G')
plt.tight_layout()
plt.show()

sys.exit()







# fmt: off
snr_list_1_string = ['0.321', '0.037', '0.030', '0.158', '0.081', '0.014', '0.129', '0.158', '0.092', '0.007', '0.053', '0.025', '0.022', '0.057', '0.102', '0.006', '0.048', '0.030', '0.078', '0.086', '0.053', '0.019', '0.070', '0.010', '0.019', '0.085', '0.105', '0.055', '0.028', '0.026', '0.071', '0.068', '0.023', '0.041', '0.014', '0.007', '0.083', '0.016', '0.118', '-0.005', '-0.001', '-0.002', '0.073', '0.104', '0.004', '-0.009', '0.027', '0.036', '0.020', '0.012', '0.045', '0.088', '0.084', '0.037', '0.018', '0.011', '0.064', '0.005', '0.019', '0.018', '0.004', '0.069', '0.096', '0.072', '0.069', '0.122', '0.012', '0.014', '0.015', '0.009', '0.013', '0.011', '0.004', '0.065', '0.086', '0.056', '0.054', '0.077', '0.021', '0.110', '0.015', '0.038', '0.013', '0.068', '0.114', '0.007', '0.065', '0.010', '0.068', '0.019', '0.026', '0.088', '0.023', '0.035', '0.028', '0.083', '0.092', '0.018', '0.022', '0.089', '-0.002', '0.089', '0.091', '0.010', '0.021', '0.082', '-0.006', '0.081', '0.004', '0.070', '0.096', '0.015', '0.070', '0.015', '0.017', '-0.007', '0.021', '0.068', '0.060', '0.067', '0.010', '0.060', '0.017', '0.014', '0.013', '0.008', '0.015', '0.083', '0.088', '0.012', '0.024', '0.007', '0.028', '0.021', '0.016', '0.010', '0.117', '-0.010', '0.019', '0.079', '0.071', '0.107', '0.016', '0.062', '0.009', '0.084', '0.106', '0.040']
snr_list_1_float = [float(d) for d in snr_list_1_string]
snr_dur_list_1 = [104, 100, 92, 84, 100, 108, 88, 104, 72, 56, 100, 272, 68, 96, 92, 132, 108, 116, 112, 76, 64, 16, 92, 144, 56, 76, 72, 48, 124, 16, 84, 64, 72, 80, 16, 44, 52, 40, 80, 116, 92, 272, 92, 72, 24, 272, 92, 56, 92, 88, 56, 68, 60, 28, 36, 36, 76, 272, 16, 64, 216, 72, 80, 116, 68, 88, 72, 84, 176, 56, 56, 68, 44, 72, 68, 48, 80, 92, 56, 84, 272, 72, 68, 80, 84, 40, 72, 236, 80, 84, 88, 88, 96, 124, 108, 80, 100, 72, 80, 64, 64, 108, 108, 140, 96, 84, 80, 72, 72, 72, 72, 144, 52, 100, 272, 28, 44, 132, 92, 100, 100, 72, 56, 40, 36, 48, 68, 72, 88, 80, 92, 272, 72, 76, 272, 16, 92, 64, 272, 72, 84, 92, 116, 100, 96, 80, 80, 128]
snr_list_1 = [0.306, 0.036, 0.04, 0.153, 0.087, 0.022, 0.138, 0.147, 0.114, 0.019, 0.05, 0.012, 0.034, 0.082, 0.128, 0.021, 0.069, 0.026, 0.073, 0.105, 0.078, 0.024, 0.073, 0.012, 0.027, 0.108, 0.107, 0.05, 0.009, 0.024, 0.075, 0.066, 0.047, 0.047, 0.018, 0.024, 0.062, 0.026, 0.14, 0.004, 0.018, 0.014, 0.079, 0.104, 0.008, 0.003, 0.029, 0.038, 0.018, 0.035, 0.071, 0.072, 0.099, 0.035, 0.042, 0, 0.045, 0.02, 0.042, 0.062, 0.023, 0.086, 0.12, 0.065, 0.042, 0.123, 0.019, 0.03, 0.01, 0.018, 0, 0.019, 0.044, 0.096, 0.099, 0.095, 0.086, 0.102, 0.019, 0.114, 0.009, 0.022, 0.038, 0.085, 0.11, 0.016, 0.09, 0.003, 0.077, 0.015, 0.046, 0.097, 0.02, 0.028, 0.039, 0.083, 0.063, 0.012, 0.026, 0.108, 0.031, 0.11, 0.072, 0.005, 0.012, 0.097, 0.014, 0.078, 0, 0.094, 0.111, 0.03, 0.072, 0.03, 0.018, 0.036, 0.013, 0.065, 0.078, 0.081, 0, 0.063, 0.03, 0.041, 0.041, 0.027, 0.015, 0.066, 0.116, 0.032, 0.021, 0.017, 0.043, 0.024, 0.008, 0.006, 0.115, 0.004, 0.011, 0.095, 0.074, 0.105, 0.015, 0.091, 0.024, 0.086, 0.101, 0.049]
snr_dur_list_2 = [272, 84, 84, 80, 100, 80, 84, 56, 56, 72, 112, 16, 68, 16, 128, 100, 84, 84, 124, 80, 176, 88, 36, 100, 72, 80, 84, 16, 84, 80, 120, 72, 92, 96, 84, 56, 272, 40, 64, 20, 72, 92, 28, 76, 76, 80, 72, 92, 80, 72, 72, 92, 16, 72, 92, 80, 76, 72, 44, 40, 72, 236, 272, 16, 272, 100, 56, 100, 72, 52, 68, 76, 64, 92, 64, 60, 76, 144, 16, 128, 64, 56, 76, 56, 48, 108, 80, 164, 272, 64, 204, 204, 100, 76, 128, 272, 76, 116, 88, 128, 80, 272, 80, 84, 100, 56, 128, 100, 36, 16, 16, 116, 16, 116, 80, 116, 68, 144, 132, 156, 16, 56, 96, 64, 84, 80, 80, 100, 84, 72, 92, 88, 88, 116, 140, 64, 152, 72, 80, 40, 64, 220, 80, 164, 164, 32, 60, 112]
snr_list_2 = [0.025, 0.098, 0.056, 0.032, 0.047, 0.113, 0.021, 0.042, 0, 0.05, 0.02, 0.015, 0.05, 0.009, 0.005, 0.084, 0.04, 0.127, 0.051, 0.01, 0.008, 0.121, 0.028, 0.093, 0.119, 0.023, 0.008, 0.037, 0.087, 0.101, 0.026, 0.046, 0.042, 0.016, 0.089, 0.05, 0.002, 0.047, 0.014, 0.737, 0.057, 0.009, 0.05, 0.017, 0.069, 0.093, 0.113, 0.012, 0.103, 0.08, 0, 0.007, 0.021, 0.056, 0.126, 0.07, 0.083, 0.134, 0.031, 0.031, 0.094, 0.001, 0.02, 0.014, 0.011, 0.015, 0.109, 0.031, 0.076, 0.023, 0.136, 0.082, 0.086, 0.013, 0.055, 0.007, 0.026, 0.021, 0.019, 0.023, 0.089, 0.037, 0.085, 0.051, 0.035, 0.103, 0.03, 0.004, 0.014, 0.053, 0.009, 0, 0.014, 0.119, 0.07, 0.009, 0.046, 0.039, 0.134, 0.012, 0.061, 0.021, 0.052, 0.092, 0.06, 0.01, 0.045, 0.011, 0.013, 0.01, 0.008, 0.132, 0.013, 0.017, 0.078, 0.021, 0.08, 0.025, 0.027, 0.045, 0.013, 0.015, 0.08, 0.078, 0.102, 0.05, 0.009, 0.005, 0.018, 0.12, 0.128, 0.081, 0, 0.007, 0.008, 0.027, 0.006, 0.018, 0.042, 0.024, 0.029, 0.014, 0.099, 0.023, 0, 0.007, 0.008, 0.096]
# fmt: on

# Ensure both lists have the same length
assert len(snr_dur_list_1) == len(snr_list_1)
assert len(snr_dur_list_2) == len(snr_list_2)
assert len(snr_list_1) == len(snr_list_2)  # Ensure matching indices

# Select the higher SNR and corresponding duration
selected_durations = [
    snr_dur_list_1[i] if snr_list_1[i] >= snr_list_2[i] else snr_dur_list_2[i]
    for i in range(len(snr_list_1))
]

selected_snrs = [max(snr_list_1[i], snr_list_2[i]) for i in range(len(snr_list_1))]


# Compute the median of valid durations (between 50 and 200)
valid_durations = [d for d in selected_durations if 50 <= d <= 200]
median_duration = round(np.median(valid_durations)) if valid_durations else 128

# Replace out-of-range durations with the median
final_durations = [d if 50 <= d <= 200 else median_duration for d in selected_durations]
print(len(final_durations))
# final_snrs = [selected_snr[d] for d in final_durations if 50 <= d <= 200]
# Compute median SNR and duration
median_snr = np.median(selected_snrs)
median_final_duration = np.median(final_durations)

# Print results
print("Selected Durations:", selected_durations)
print("Final Durations:", final_durations)
print("Final SNRs:", selected_snrs)
print("Median SNR:", round(median_snr, 3))
print("Median Duration:", round(median_final_duration, 1))


# remove outlies
def remove_outliers(snr_list):
    Q1 = np.percentile(snr_list, 25)
    Q3 = np.percentile(snr_list, 75)
    median = np.median(snr_list)
    IQR = Q3 - Q1
    lower_bound = median - 2 * IQR
    upper_bound = median + 5 * IQR
    selected_inds = [
        ind
        for ind in range(len(snr_list))
        if lower_bound <= snr_list[ind] <= upper_bound
    ]
    return selected_inds


selected_inds = remove_outliers(selected_snrs)
filtered_durations = [final_durations[ind] for ind in selected_inds]
filtered_snrs = [selected_snrs[ind] for ind in selected_inds]
filtered_median_duration = np.median(filtered_durations)
filtered_median_snr = np.median(filtered_snrs)
kpl.init_kplotlib()
plt.figure(figsize=(6, 5))
plt.scatter(
    filtered_durations,
    filtered_snrs,
    marker="o",
    color="blue",
    edgecolors="black",
    alpha=0.6,
)
plt.title("SCC Durations vs. SNRs")
plt.xlabel("Optimized SCC duration (ns)")
plt.ylabel("SNR")
# Add text at the top-right of the figure
plt.figtext(
    0.97,
    0.9,  # (x, y) position in figure coordinates (1 = right/top, 0 = left/bottom)
    f"Med. SCC Duration: {filtered_median_duration:.2f} ns\nMed. SCC SNR: {filtered_median_snr:.2f}",
    fontsize=11,
    ha="right",
    va="top",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)
# plt.show(block=True)


import numpy as np

nv_ind_include = []
# fmt: off
nv_ind_69MHz= [3, 7, 8, 13, 22, 25, 30, 38, 42, 50, 51, 58, 61, 76, 77, 79, 84, 88, 92, 96, 101, 105, 107, 109, 119, 128, 132, 133, 134, 139, 140, 141, 143, 145]
nv_ind_178MHz = [0, 4, 6, 14, 19, 20, 26, 31, 33, 36, 39, 43, 52, 59, 62, 63, 64, 65, 74, 75, 78, 83, 86, 90, 91, 95, 99, 110, 112, 113, 121, 126, 127, 136, 146]
# fmt: on
nv_ind_include = nv_ind_69MHz + nv_ind_178MHz
print(nv_ind_include)
nv_ind_include = sorted(nv_ind_include)

print(f"sorted nv indices = {nv_ind_include}")


from utils import kplotlib as kpl

kpl.init_kplotlib()
# phase = np.load("slmsuite\computed_phase\slm_phase_148nvs_20250203_171815.npy")

# Load phase data
phase = np.load(r"slmsuite\computed_phase\slm_phase_148nvs_20250203_171815.npy")
phase = np.load(r"slmsuite\computed_phase\slm_phase_117nvs_20241230_162649.npy")

# # Plot phase data
# plt.figure(figsize=(8, 4.2))
# plt.imshow(phase)  # Use 'jet' or another colormap
# plt.colorbar(label="Phase (radians)", pad=0.01)  # Optional: adds a colorbar
# plt.title("SLM Phase")
# plt.xticks([])
# plt.yticks([])
# plt.show(block=True)


# Plot phase data
fig = plt.figure(figsize=(7, 4))
# im = plt.imshow(phase)  # Proper color scale
im = plt.imshow(phase, cmap="twilight", vmin=0, vmax=2 * np.pi)  # Proper color scale

# Add colorbar with correct label
cbar = plt.colorbar(im, pad=0.01, shrink=0.74)
cbar.set_label("Phase (radians)", fontsize=15)
cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"], fontsize=15)

# Set labels and title
plt.xlabel("SLM X Pixels", fontsize=15)  # X-axis corresponds to SLM columns
plt.ylabel("SLM Y Pixels", fontsize=15)  # Y-axis corresponds to SLM rows
plt.title("SLM Phase Pattern", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.subplots_adjust(left=0.001, right=0.9, bottom=0.1, top=0.9)
plt.show()


# Load the image data
image_path = r"slmsuite\cam_image\slm_generated_spots_117nvs_20250224_150218.npy"
spots_image = np.load(image_path)

# Flip and rotate the image
# spots_image = np.flip(spots_image)  # Flip vertically and horizontally
spots_image = np.rot90(spots_image, k=1)  # Rotate by 90 degrees counterclockwise
spots_image = spots_image[330:1200, 100:1000]
# Set intensity limits
vmin = np.min(spots_image)
vmax = 20  # Set the max intensity limit

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Display the image with white-to-orange color mapping
im = ax.imshow(
    spots_image,
    cmap="Oranges",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)

# Add a colorbar

# Add colorbar with set limits
cbar = fig.colorbar(im, pad=0.02)
cbar.set_label("Intensity (a.u.)", color="black")
cbar.ax.yaxis.set_tick_params(color="black")  # Colorbar ticks in black
cbar.ax.yaxis.set_tick_params(labelcolor="black")  # Tick labels in black

# Set axis labels and title with black text
ax.set_xlabel("Camera X Pixels", color="black")
ax.set_ylabel("Camera Y Pixels", color="black")
ax.set_title("SLM Generated Spots", color="black")

# Adjust ticks to be visible on white background
ax.tick_params(axis="both", colors="black")

# Show the plot
kpl.show()

plt.show(block=True)
