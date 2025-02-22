import numpy as np
from utils import kplotlib as kpl
from matplotlib import pyplot as plt

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
plt.show(block=True)

nv_ind_include = []
# fmt: off
nv_ind_69MHz= [3, 7, 8, 13, 22, 25, 30, 38, 42, 50, 51, 58, 61, 76, 77, 79, 84, 88, 92, 96, 101, 105, 107, 109, 119, 128, 132, 133, 134, 139, 140, 141, 143, 145]
nv_ind_178MHz = [0, 4, 6, 14, 19, 20, 26, 31, 33, 36, 39, 43, 52, 59, 62, 63, 64, 65, 74, 75, 78, 83, 86, 90, 91, 95, 99, 110, 112, 113, 121, 126, 127, 136, 146]
# fmt: on
nv_ind_include = nv_ind_69MHz + nv_ind_178MHz
print(nv_ind_include)
nv_ind_include = sorted(nv_ind_include)

print(f"sorted nv indices = {nv_ind_include}")
