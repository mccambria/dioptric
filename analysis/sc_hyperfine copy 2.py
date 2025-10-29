import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
from pathlib import Path
import sys
from utils import kplotlib as kpl

# kpl.init_kplotlib()


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


# Main simulation
data = load_hyperfine_data()
A_zz = data["Azz"].values[:10]
# A_zz = data["Azz"].values
print(A_zz)
taus = np.linspace(0, 600e-6, 121)
# Monte Carlo parameters
N_trials = 10000
corr_time = 1e-3  # ~1 ms nuclear flip correlation time
gamma = 175.6  # flip rate (s^-1) fitted to give correct T2

# Assume your dataframe has 'x', 'y', 'z', 'Azz' columns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot_hyperfine_map(data, top_n=10, color_by="Azz"):
    """
    Plots a 3D scatter map of nuclear spins colored by a chosen hyperfine parameter (default Azz).

    Parameters:
        data (pd.DataFrame): Must contain 'x', 'y', 'z', and 'Azz' columns.
        top_n (int): Number of top nuclear spin sites to include (by absolute Azz).
        color_by (str): Column to color-code by. Must be in data (e.g., 'Azz', 'distance').
    """
    if color_by not in data.columns:
        raise ValueError(f"{color_by} not in dataframe columns.")

    # Select top_n sites by |Azz| strength
    subset = data.copy()
    subset["Azz_abs"] = subset["Azz"].abs()
    subset = subset.sort_values(by="Azz_abs", ascending=False).head(top_n)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        subset["x"], subset["y"], subset["z"], c=subset[color_by], cmap="viridis", s=25
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"Top {top_n} Nuclear Spins by |Azz|")
    fig.colorbar(scatter, label=f"{color_by} (MHz)")
    plt.tight_layout()
    kpl.show(block=True)


def fancy_hyperfine_plot(
    data, top_n=100, color_by="Azz", cmap="plasma", scale_size=True
):
    """
    Fancy 3D plot of nuclear spins showing hyperfine interactions.

    Parameters:
        data (pd.DataFrame): Must contain 'x', 'y', 'z', and hyperfine parameter columns.
        top_n (int): Number of points to show, ranked by |Azz|.
        color_by (str): Column to color points by (e.g., 'Azz').
        cmap (str): Matplotlib colormap for visualization.
        scale_size (bool): Whether to scale marker size by hyperfine magnitude.
    """
    if color_by not in data.columns:
        raise ValueError(f"'{color_by}' not found in data columns")

    df = data.copy()
    df["Azz_abs"] = df["Azz"].abs()
    df = df.sort_values(by="Azz_abs", ascending=False).head(top_n)

    # Marker size scales
    if scale_size:
        size = np.interp(
            df["Azz_abs"], (df["Azz_abs"].min(), df["Azz_abs"].max()), (30, 300)
        )
    else:
        size = 60

    # Plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        df["x"],
        df["y"],
        df["z"],
        c=df[color_by],
        cmap=cmap,
        s=size,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )

    # Highlight NV center
    ax.scatter(
        0,
        0,
        0,
        c="red",
        s=400,
        marker="*",
        edgecolors="black",
        label="NV Center",
        zorder=10,
    )

    # Aesthetics
    ax.set_title(f"Top {top_n} Spins by |{color_by}|", fontsize=16, pad=20)
    ax.set_xlabel("X (Å)", labelpad=10)
    ax.set_ylabel("Y (Å)", labelpad=10)
    ax.set_zlabel("Z (Å)", labelpad=10)
    # ax.grid(True, linestyle=":", linewidth=0.5)

    # Equal aspect ratio
    ranges = [
        df["x"].max() - df["x"].min(),
        df["y"].max() - df["y"].min(),
        df["z"].max() - df["z"].min(),
    ]
    max_range = max(ranges) / 2.0
    mid_x = (df["x"].max() + df["x"].min()) * 0.5
    mid_y = (df["y"].max() + df["y"].min()) * 0.5
    mid_z = (df["z"].max() + df["z"].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # View angle
    ax.view_init(elev=25, azim=40)

    # Colorbar
    cbar = fig.colorbar(p, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label(f"{color_by} (MHz)", fontsize=12)

    ax.legend(loc="upper left")
    plt.tight_layout()


fancy_hyperfine_plot(data, top_n=1500, color_by="Azz")
kpl.show(block=True)
# fancy_hyperfine_plot(data, top_n=150, color_by="Azz", cmap="plasma")

# plot_hyperfine_map(data, top_n=150, color_by="Azz")  # top 150 sites
# plot_hyperfine_map(
#     data, top_n=50, color_by="distance"
# )  # color by radial distance instead


def simulate_semiclassical_echo_with_B(
    A_tensors, taus, Bz=0.00475, gamma=175.6, N_trials=10000
):
    A_zz = np.array([A[2, 2] for A in A_tensors])
    N = len(A_zz)
    gamma_n = 10.705e6  # Hz/T for 13C
    omega_n = gamma_n * Bz * 2 * np.pi  # Convert to rad/s

    coherence = []
    for tau in taus:
        samples = []
        for _ in range(N_trials):
            phase = 0
            for k in range(N):
                I_z = np.random.choice([+0.5, -0.5])
                flip = np.random.rand() < 1 - np.exp(-gamma * tau)
                if flip:
                    # Echo phase: includes both hyperfine and Zeeman phase shifts
                    delta_phi = -(A_zz[k] + omega_n) * I_z * tau
                    phase += delta_phi
            samples.append(np.cos(phase))
        coherence.append(np.mean(samples))
    return np.array(coherence)


# echo_signal = simulate_semiclassical_echo_with_B(
#     A_zz, taus, Bz=4.75e-3, gamma=175.6, N_trials=10000
# )
# plot_echo(taus, echo_signal)

kpl.show(block=True)
