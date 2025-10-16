import matplotlib.pyplot as plt
import numpy as np

# # Simulation parameters
# n_shots = 1000  # Number of repetitions per state
# sigma = 0.5  # Noise standard deviation
# iq0 = (1.0, 0.0)  # Ideal IQ point for |0⟩
# iq1 = (-1.0, 0.0)  # Ideal IQ point for |1⟩

# # Generate noisy samples around each IQ point
# I0 = np.random.normal(loc=iq0[0], scale=sigma, size=n_shots)
# Q0 = np.random.normal(loc=iq0[1], scale=sigma, size=n_shots)

# I1 = np.random.normal(loc=iq1[0], scale=sigma, size=n_shots)
# Q1 = np.random.normal(loc=iq1[1], scale=sigma, size=n_shots)

# # Plot IQ blobs
# plt.figure(figsize=(6, 5))
# plt.scatter(I0, Q0, color="blue", label="|0⟩", alpha=0.5)
# plt.scatter(I1, Q1, color="red", label="|1⟩", alpha=0.5)
# plt.axvline(x=0, color="gray", linestyle="--", label="I threshold")
# plt.xlabel("I")
# plt.ylabel("Q")
# plt.title("Simulated Qubit Readout IQ Blobs")
# plt.legend()
# plt.axis("equal")
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- Sim params --------------------
rng = np.random.default_rng(7)
phases = np.linspace(0, 360, 361)  # degrees

def inverted_fringe(phase_deg, visibility):
    # Min at 0°, max at 180° (cosine inverted)
    return 0.5 * (1 - visibility * np.cos(np.radians(phase_deg)))

# Visibilities (SE < XY8-8 < XY8-16)
V_SE = 0.55
V_XY8_8 = 0.80
V_XY8_16 = 0.90

# Ideal signals
se_ideal = inverted_fringe(phases, V_SE)
xy8_8_ideal = inverted_fringe(phases, V_XY8_8)
xy8_16_ideal = inverted_fringe(phases, V_XY8_16)

# Add optional realistic noise (toggle on/off)
noise_on = True
noise_sd = 0.010
se_noisy = se_ideal + (rng.normal(0, noise_sd, size=se_ideal.shape) if noise_on else 0)
xy8_8_noisy = xy8_8_ideal + (rng.normal(0, noise_sd, size=xy8_8_ideal.shape) if noise_on else 0)
xy8_16_noisy = xy8_16_ideal + (rng.normal(0, noise_sd, size=xy8_16_ideal.shape) if noise_on else 0)

# -------------------- Plot (single figure) --------------------
plt.figure(figsize=(9, 6))
plt.plot(phases, se_noisy, label="Spin Echo (SE)")
plt.plot(phases, xy8_8_noisy, label="XY8-8")
plt.plot(phases, xy8_16_noisy, label="XY8-16")
plt.axvline(90, linestyle="--", label="Linear region (~90°)")
plt.xlabel("Final pi/2 pulse phase (degrees)")
plt.ylabel("Normalized counts (ms=0)")
plt.title("Phase-Sweep Fringes (SE vs XY8) — min at 0°, max at 180°")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

# png_path = "/mnt/data/fringes_SE_XY8_overlay.png"
# plt.savefig(png_path, dpi=200)
plt.show()

# -------------------- Save data table --------------------
# df = pd.DataFrame({
#     "phase_deg": phases,
#     "SE_noisy": se_noisy,
#     "XY8_8_noisy": xy8_8_noisy,
#     "XY8_16_noisy": xy8_16_noisy,
#     "SE_ideal": se_ideal,
#     "XY8_8_ideal": xy8_8_ideal,
#     "XY8_16_ideal": xy8_16_ideal
# })
# csv_path = "/mnt/data/fringes_SE_XY8_overlay.csv"
# df.to_csv(csv_path, index=False)

# import caas_jupyter_tools as cj
# cj.display_dataframe_to_user("Fringe comparison data (SE vs XY8)", df)  # interactive table
