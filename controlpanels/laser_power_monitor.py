import nidaqmx
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Config
channels = {"520nm": "Dev1/ai0", "589nm": "Dev1/ai1", "638nm": "Dev1/ai2"}
read_interval = 5  # seconds (or 15*60 for 15 minutes)

# Create timestamped CSV file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"multi_laser_log_{timestamp}.csv"
csv_file = open(filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time (min)"] + list(channels.keys()))

# Initialize plotting
plt.ion()
fig, ax = plt.subplots()
time_vals = []
laser_vals = {name: [] for name in channels}
lines = {name: ax.plot([], [], label=name)[0] for name in channels}
ax.set_xlabel("Time (min)")
ax.set_ylabel("Photodiode Voltage (V)")
ax.legend()

# Start reading
start_time = time.time()

try:
    with nidaqmx.Task() as task:
        for ch in channels.values():
            task.ai_channels.add_ai_voltage_chan(ch)

        while True:
            now = time.time()
            t_min = (now - start_time) / 60
            readings = task.read()
            if isinstance(readings, float):
                readings = [readings]  # handle single-channel case

            # Log and update
            time_vals.append(t_min)
            row = [t_min]
            for idx, name in enumerate(channels):
                val = readings[idx]
                laser_vals[name].append(val)
                row.append(val)

                # Update plot
                lines[name].set_xdata(time_vals)
                lines[name].set_ydata(laser_vals[name])

            # Write to CSV
            csv_writer.writerow(row)

            # Update plot
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)

            print(
                f"[{t_min:.2f} min] "
                + ", ".join(
                    f"{k}: {v:.4f} V" for k, v in zip(channels.keys(), readings)
                )
            )
            time.sleep(read_interval)

except KeyboardInterrupt:
    print("Interrupted. Saving data...")
    csv_file.close()
