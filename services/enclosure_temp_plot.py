import datetime
import os
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Base folder and current month-year folder
base_folder = "G:\\Enclosure_Temp"
folder = datetime.datetime.now().strftime("%m%Y")
data_folder = os.path.join(base_folder, folder)

# Define channels and corresponding filenames
channels = {
    "4A": "temp_4A.csv",
    "4B": "temp_4B.csv",
    "4C": "temp_4C.csv",
}

# Live plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))


def update_plot():
    ax.clear()
    now = datetime.datetime.now()
    past_24h = now - datetime.timedelta(hours=24)

    for label, filename in channels.items():
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path):
            print(f"Waiting for file: {filename}")
            continue

        try:
            # Read and parse CSV
            df = pd.read_csv(
                file_path,
                names=["Timestamp", "Temperature"],
                parse_dates=["Timestamp"],
                dtype={"Temperature": float},
            )

            # Filter last 24 hours
            df = df[df["Timestamp"] > past_24h]

            # Plot
            ax.plot(df["Timestamp"], df["Temperature"], label=f"Channel {label}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    ax.set_title("Live Temperature Plot (Last 24h)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature [°C]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend()
    fig.autofmt_xdate()
    plt.pause(0.1)


# def main():
#     print(f"Live plotting from: {data_folder}")
#     while True:
#         update_plot()
#         time.sleep(15 * 60)  # Refresh every minute
#         input("press enter  ...")


# def main():
#     print(f"Live plotting from: {data_folder}")
#     try:
#         while True:
#             update_plot()
#             time.sleep(15 * 60)
#     except KeyboardInterrupt:
#         print("Live plotting stopped by user.")
#         plt.close()
def main():
    print(f"Live plotting from: {data_folder}")
    try:
        while True:
            update_plot()
            if (
                input("Press Enter to rephresh or type 'q' to quit: ").strip().lower()
                == "q"
            ):
                break
    finally:
        print("Exiting and closing plot.")
        plt.close()


if __name__ == "__main__":
    main()
