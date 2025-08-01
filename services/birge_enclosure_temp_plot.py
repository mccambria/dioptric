import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import kplotlib as kplt

kplt.init_kplotlib()
# === 1. Load and parse the CSV ===
file_path = "G:\\NV_Widefield_RT_Setup_Enclosure_Temp_Logs\\birge_temp_stick_data\\TS00NAHQ2A_2025_07_10.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)
# print(df.keys())
print(df.columns.tolist())

# === 2. Parse datetime and extract date ===
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Date"] = df["Timestamp"].dt.date

# === 3. Basic plot: Full temperature time-series ===
plt.figure(figsize=(12, 4))
plt.plot(df["Timestamp"], df["Temperature"], label="Temperature (°C)")
plt.title("Birge 241 Temperature (May 9–19)", fontsize=15)
plt.xlabel("Timestamp", fontsize=15)
plt.ylabel("Temperature (°C)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.legend()
plt.show()

# === 4. Group by day and compute stats ===
daily_stats = (
    df.groupby("Date")["Temperature"].agg(["mean", "std", "min", "max"]).reset_index()
)

# === 5. Plot daily average temperature with error bars ===
plt.figure(figsize=(10, 5))
plt.errorbar(
    daily_stats["Date"],
    daily_stats["mean"],
    yerr=daily_stats["std"],
    fmt="-o",
    capsize=5,
)
plt.title("Daily Mean Temperature with Standard Deviation", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.ylabel("Temperature (°C)", fontsize=15)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.show()

# === 6. Optional: Boxplot for per-day temperature distribution ===
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Date", y="Temperature")
plt.title("Daily Temperature Distribution", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.show(block=True)
