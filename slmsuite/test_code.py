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

import numpy as np
import matplotlib.pyplot as plt

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

# fmt: off
# Extended data for "Counts at initial coordinates" and "Counts after drift compensation"
extended_initial_counts = [
    1200.0, 1267.0, 1189.0, 1179.0, 1203.0, 1287.0, 1170.0, 1155.0, 1259.0, 1168.0,
    1320.0, 1361.0, 1335.0, 1337.0, 1345.0, 1435.0, 1324.0, 1369.0, 1358.0, 1385.0,
    1322.0, 1325.0, 1307.0, 1286.0, 1340.0, 1334.0, 1344.0, 1317.0, 1327.0, 1272.0,
    1323.0, 1336.0, 1373.0, 1406.0, 1349.0, 1364.0, 1425.0, 1359.0, 1376.0, 1362.0,
    1392.0, 1359.0, 1369.0, 1402.0, 1354.0, 1435.0, 1457.0, 1416.0, 1398.0, 1362.0,
    1385.0, 1365.0, 1382.0, 1410.0, 1371.0, 1347.0, 1385.0, 1405.0, 1443.0, 1462.0,
    1355.0, 1382.0, 1406.0, 1431.0, 1352.0, 1396.0, 1452.0, 1379.0, 1420.0, 1344.0,
    1387.0, 1405.0, 1400.0, 1468.0, 1359.0, 1438.0, 1425.0, 1444.0, 1417.0, 1430.0,
    1350.0, 1439.0, 1419.0, 1431.0, 1421.0, 1402.0, 1324.0, 1354.0
]
extended_compensated_counts = [
    1303.0, 1287.0, 1301.0, 1254.0, 1220.0, 1254.0, 1226.0, 1309.0, 1250.0, 1269.0,
    1272.0, 1347.0, 1303.0, 1362.0, 1360.0, 1290.0, 1382.0, 1328.0, 1322.0, 1274.0,
    1340.0, 1301.0, 1357.0, 1376.0, 1391.0, 1388.0, 1360.0, 1370.0, 1323.0, 1370.0,
    1366.0, 1378.0, 1344.0, 1371.0, 1320.0, 1309.0, 1362.0, 1319.0, 1326.0, 1361.0,
    1365.0, 1378.0, 1361.0, 1291.0, 1366.0, 1354.0, 1374.0, 1372.0, 1276.0, 1336.0,
    1389.0, 1386.0, 1360.0, 1366.0, 1295.0, 1278.0, 1344.0, 1316.0, 1376.0, 1365.0,
    1336.0, 1380.0, 1307.0, 1305.0, 1316.0, 1333.0, 1423.0, 1351.0, 1353.0, 1341.0,
    1405.0, 1347.0, 1282.0, 1352.0, 1371.0, 1336.0, 1370.0, 1406.0, 1399.0, 1295.0,
    1343.0, 1428.0, 1406.0, 1308.0, 1364.0, 1332.0, 1340.0, 1383.0
]
# fmt: on
# Combine all counts into one list
combined_counts_all = extended_initial_counts + extended_compensated_counts
print(len(combined_counts_all))
print(np.sqrt(len(combined_counts_all)))
# Calculate updated statistical measures
plt.figure(figsize=(8, 6))
plt.hist(combined_counts_all, bins=15, color="blue", alpha=0.7)
plt.title("Combined Counts Distribution", fontsize=14)
plt.xlabel("Counts", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(alpha=0.3)
plt.show()
