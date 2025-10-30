import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
import scipy.linalg as la
import pandas as pd
from pathlib import Path
import sys
import os
from joblib import Parallel, delayed
from datetime import datetime

kpl.init_kplotlib()
# Constants
D = 2 * np.pi * 2.87e9  # NV zero-field splitting in rad/s
gamma_e = 1.760859644e11  # Electron gyromagnetic ratio (rad/s/T)
gamma_C = 6.728284e7  # 13C gyromagnetic ratio (rad/s/T)
B = 47.5e-4  # Tesla

# Pauli matrices
sigx = np.array([[0, 1], [1, 0]], dtype=complex)
sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigz = np.array([[1, 0], [0, -1]], dtype=complex)


# Load hyperfine tensors (Axx...Ayz) from file
def load_hyperfine_data():
    file_path = Path(r"analysis/nv_hyperfine_coupling/nv-2.txt")
    with open(file_path, "r") as f:
        lines = f.readlines()
    data_start = next(
        i for i, line in enumerate(lines) if line.strip().startswith("1 ")
    )
    data = pd.read_csv(
        file_path,
        sep="\s+",
        skiprows=data_start,
        header=None,
        names=[
            "index",
            "distance",
            "x",
            "y",
            "z",
            "Axx",
            "Ayy",
            "Azz",
            "Axy",
            "Axz",
            "Ayz",
        ],
    )
    return data


def save_results(coherence, index, A_zz, filename):
    # Ensure the directory exists
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it doesn't exist

    # Save the data to a .npz file
    np.savez(
        filename,
        coherence=coherence,
        taus=taus,
        index=index,
        A_zz=A_zz,
    )


def load_data(
    file_path="analysis\nv_hyperfine_coupling\simulated_coherence_220_sites.npz",
):
    data = np.load(file_path)
    print(data.keys())
    coherence = data["coherence"]
    taus = data["taus"]
    index = data["index"]
    A_zz = data["A_zz"]
    return (
        coherence,
        taus,
        index,
        A_zz,
    )


# Plot results
def plot_echo(taus, signal, title="Echo coherence"):
    plt.figure(figsize=(8, 4))
    plt.plot(taus * 1e6, signal, label="Semiclassical")
    plt.xlabel("Echo delay $\\tau$ ($\\mu$s)")
    plt.ylabel("Coherence")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# # Simulate echo decay using semiclassical method
# def simulate_semiclassical_echo(data, taus, n, gamma=175.6, N_trials=10000):
#     A_zz = data["Azz"].values[:n]
#     index = data["index"].values[:n]
#     N = len(A_zz)
#     coherence = []
#     for tau in taus:
#         samples = []
#         for _ in range(N_trials):
#             phase = 0
#             for k in range(N):
#                 I_z = np.random.choice([+0.5, -0.5])
#                 flip = np.random.rand() < 1 - np.exp(-gamma * tau)
#                 if flip:
#                     phase += -A_zz[k] * I_z * tau
#             samples.append(np.cos(phase))
#         coherence.append(np.mean(samples))

#     save_results(
#         coherence,
#         taus,
#         index,
#         A_zz,
#         filename=rf"analysis\nv_hyperfine_coupling\simulated_coherence_{len(index)}.npz",
#     )
#     return np.array(coherence)

# === PARAMETERS ===
gamma = 175.6  # nuclear flip rate (1/s)
N_trials = 10000
taus = np.linspace(0, 600e-5, 121)  # echo delays (seconds)
spin_block_size = 6
max_spin = 60  # total spins to consider
save_dir = Path("analysis/nv_hyperfine_coupling")
save_dir.mkdir(parents=True, exist_ok=True)

data = load_hyperfine_data()

# distance =data["distance"].values
# mask = distance < 15 
# print(len(distance[mask]))
# sys.exit()
# === SEMICLASSICAL SIMULATION ===
def simulate_semiclassical_echo(index_range, taus, gamma=175.6, N_trials=1000):
    A_zz = data["Azz"].values[list(index_range)]
    coherence = []
    for tau in taus:
        samples = []
        for _ in range(N_trials):
            phase = sum(
                (
                    -A * np.random.choice([+0.5, -0.5]) * tau
                    if np.random.rand() < 1 - np.exp(-gamma * tau)
                    else 0
                )
                for A in A_zz
            )
            samples.append(np.cos(phase))
        coherence.append(np.mean(samples))
    return np.array(coherence)


# Define spin ranges
spin_ranges = [
    range(i, i + spin_block_size) for i in range(0, max_spin, spin_block_size)
]

# Run in parallel
results = Parallel(n_jobs=-1)(
    delayed(simulate_semiclassical_echo)(spin_range, taus, gamma, N_trials)
    for spin_range in spin_ranges
)

# === COMBINE RESULTS ===
echo_total = np.prod(results, axis=0)
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# === SAVE ===
# np.savez(
#     save_dir / f"semiclassical_clusters_{max_spin}_spins_{now}.npz",
#     taus=taus,
#     echo_total=echo_total,
#     cluster_echoes=np.array(results),
#     cluster_ranges=np.array(spin_ranges),
# )
# echo_total = np.round(echo_total, 3)
# === PLOT ===
plt.figure(figsize=(8, 6))
plt.plot(taus * 1e6, echo_total, color="black", lw=2, label="Total Echo")

# for i, cluster in enumerate(results):
#     plt.plot(taus * 1e6, cluster, "--", alpha=0.5, label=f"Cluster {i+1}")

plt.xlabel("Echo delay $\\tau$ (μs)")
plt.ylabel("Coherence")
plt.title("Semiclassical NV Echo with Clustered Spin Bath")
plt.grid(True)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
# plt.savefig(save_dir / f"semiclassical_clusters_{max_spin}_spins.png", dpi=300)
plt.show(block=True)


sys.exit()


# def plot_semiclassical_vs_distance_cutoffs(
#     distances, Azz_full, taus, gamma=175.6, N_trials=2000, cutoffs=[10, 15, 20, 25, 30]
# ):

#     plt.figure(figsize=(10, 5))
#     for cutoff in cutoffs:
#         mask = distances > cutoff
#         A_zz_cut = Azz_full[mask]
#         coh = simulate_semiclassical_echo(
#             A_zz_cut * 2 * np.pi * 1e6, taus, gamma, N_trials
#         )
#         plt.plot(taus * 1e6, coh, label=f"r > {cutoff} Å (n={len(A_zz_cut)})")

#     plt.xlabel("Echo delay $\\tau$ ($\\mu$s)")
#     plt.ylabel("Semiclassical coherence")
#     plt.title("Semiclassical Decoherence vs. Distance Cutoff")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# distances = data["distance"].values

# Azz = data["Azz"].values
# taus = np.linspace(0, 200e-6, 121)
# plot_semiclassical_vs_distance_cutoffs(distances, Azz, taus)

# sys.exit()


# def simulate_semiclassical_echo_with_B(
#     A_tensors, taus, Bz=0.00475, gamma=175.6, N_trials=10000
# ):
#     A_zz = np.array([A[2, 2] for A in A_tensors])
#     N = len(A_zz)
#     gamma_n = 10.705e6  # Hz/T for 13C
#     omega_n = gamma_n * Bz * 2 * np.pi  # Convert to rad/s

#     coherence = []
#     for tau in taus:
#         samples = []
#         for _ in range(N_trials):
#             phase = 0
#             for k in range(N):
#                 I_z = np.random.choice([+0.5, -0.5])
#                 flip = np.random.rand() < 1 - np.exp(-gamma * tau)
#                 if flip:
#                     # Echo phase: includes both hyperfine and Zeeman phase shifts
#                     delta_phi = -(A_zz[k] + omega_n) * I_z * tau
#                     phase += delta_phi
#             samples.append(np.cos(phase))
#         coherence.append(np.mean(samples))
#     return np.array(coherence)


# echo_signal = simulate_semiclassical_echo_with_B(
#     A_zz, taus, Bz=4.75e-3, gamma=175.6, N_trials=10000
# )
# plot_echo(taus, echo_signal)
