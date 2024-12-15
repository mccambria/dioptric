import matplotlib.pyplot as plt
import numpy as np

# Define parameters
dead_time = 12  # Dead time in ms
readout_times = np.arange(12, 121, 6)  # Range of readout times from 12 to 120 ms

# Calculate cycle times and efficiency
cycle_times = readout_times + dead_time
efficiencies = readout_times / cycle_times
measurement_rates = 1000 / cycle_times  # Measurements per second

# Base SNR for minimum readout time
base_snr = 0.1  # Arbitrary base SNR for 12 ms
# snr_values = base_snr * np.sqrt(readout_times / 12)  # SNR scales as sqrt(T_readout)
relative_snr = base_snr * np.sqrt(readout_times / 12)  # Relative improvement in SNR


def calculate_best_readout_time(
    dead_time, readout_times, efficiencies, measurement_rates
):
    # Set thresholds for efficiency and measurement rate
    efficiency_threshold = 0.67  # Minimum 67% efficiency
    measurement_rate_threshold = 15  # Minimum 10 measurements per second

    # Find the optimal readout time based on thresholds
    optimal_time = None
    for rt, eff, rate in zip(readout_times, efficiencies, measurement_rates):
        if eff >= efficiency_threshold and rate >= measurement_rate_threshold:
            optimal_time = rt
            break

    return optimal_time


def create_plot(optimal_time):
    plt.figure(figsize=(10, 5))
    # Combine all metrics
    plt.subplot(1, 2, 1)
    plt.plot(readout_times, efficiencies * 100, marker="o", label="Efficiency (%)")
    plt.plot(readout_times, cycle_times, marker="s", label="Cycle Time (ms)")
    plt.plot(
        readout_times, measurement_rates, marker="^", label="Measurement Rate (Hz)"
    )
    plt.axvline(
        optimal_time,
        color="g",
        linestyle="--",
        label=f"Optimal Time = {optimal_time} ms",
    )
    plt.title("Combined Metrics")
    plt.xlabel("Readout Time (ms)")
    plt.ylabel("Metrics")
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(readout_times, relative_snr, marker="^", label="SNR")
    plt.title("SNR vs Readout Time")
    plt.xlabel("Readout Time (ms)")
    plt.ylabel("Relative SNR")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# Determine the optimal readout time
optimal_readout_time = calculate_best_readout_time(
    dead_time, readout_times, efficiencies, measurement_rates
)
print(f"Optimal Readout Time: {optimal_readout_time} ms")

# Generate plots with optimal time highlighted
create_plot(optimal_readout_time)
