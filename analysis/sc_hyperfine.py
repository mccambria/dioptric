import numpy as np
import numpy.linalg as LA
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit

# Physical constants
mu0 = 4 * np.pi * 1e-7  # vacuum permeability (H/m)
hbar = 1.054571817e-34  # Planck's constant / 2π (J·s)
h = 2 * np.pi * hbar
gamma_e = 1.760859644e11  # electron gyromagnetic ratio (rad/s/T)
gamma_C = 6.728284e7  # 13C gyromag. ratio (rad/s/T)
B = 47.5e-4  # 47.5 G in Tesla

# Larmor frequency of 13C at 47.5 G
omega_L = gamma_C * B  # (rad/s)
f_L = omega_L / (2 * np.pi)  # Larmor in Hz
f_L_MHz = f_L / 1e6  # Larmor in MHz
print(f"13C Larmor freq ~ {f_L_MHz:.3f} MHz")

# Spin-1/2 Pauli matrices and spin operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I_half = np.eye(2, dtype=complex)
# Spin-1/2 operators
I_x_single = 0.5 * sigma_x
I_y_single = 0.5 * sigma_y
I_z_single = 0.5 * sigma_z

# Load data from the uploaded file
file_path = Path(r"analysis\nv_hyperfine_coupling\nv-2.txt")

# Read lines to find where the actual data begins
with open(file_path, "r") as f:
    lines = f.readlines()

data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("1 "))

# Load data as DataFrame
data = pd.read_csv(
    file_path,
    delim_whitespace=True,
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

# Extract coordinates and tensor components
coords_all = data[["x", "y", "z"]].values
A_components_all = data[["Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz"]].values


# Function 1: retrieve sites by explicit index list (as before)
def get_nv_hyperfine_by_index(indices):
    coords = coords_all[indices]
    tensors = A_components_all[indices]
    return coords, tensors


# Function 2: retrieve all sites within a distance threshold (in Å)
def get_nv_hyperfine_within_distance(max_distance):
    mask = data["distance"] <= max_distance
    indices = data[mask].index.to_list()
    coords = coords_all[indices]
    tensors = A_components_all[indices]
    return indices, coords, tensors


# Function 3: retrieve all sites up to a given row number (index-based)
def get_nv_hyperfine_until_index(max_index):
    indices = list(range(max_index + 1))
    coords = coords_all[indices]
    tensors = A_components_all[indices]
    return indices, coords, tensors


# Example usage
indices = [1, 5, 10, 15, 20]  # specify desired indices here
# indices = data.nlargest(10, "Azz").index.to_list()  ## Top 10 by Azz
coords_A, A_tensors = get_nv_hyperfine_by_index(indices)

# Sites within 5 Å
# indices, coords_A, A_tensors = get_nv_hyperfine_within_distance(4.0)
#
# First 100 sites
# indices, coords_A, A_tensors = get_nv_hyperfine_until_index(99)

# Return summary
print(f"Number of 13C sites selected: {len(indices)}")
print(coords_A)
print(A_tensors)

N = len(A_tensors)  # number of nuclear spins
dim_nuc = 2**N  # nuclear Hilbert space dimension

# Construct nuclear spin operators for each spin i in the full nuclear space
I_x = []
I_y = []
I_z = []
for i in range(N):
    # identity on all spins before and after spin i
    I_before = np.eye(2**i, dtype=complex)
    I_after = np.eye(2 ** (N - i - 1), dtype=complex)
    # operator on spin i (Kron with identities)
    I_x.append(np.kron(np.kron(I_before, I_x_single), I_after))
    I_y.append(np.kron(np.kron(I_before, I_y_single), I_after))
    I_z.append(np.kron(np.kron(I_before, I_z_single), I_after))

# Assemble the Hamiltonian H = H_nuclear Zeeman + H_hyperfine + H_dipolar (in MHz units)
H_nuc = np.zeros((dim_nuc, dim_nuc), dtype=complex)
# Nuclear Zeeman: -f_L * sum I_z (negative sign so spin up is lower energy)
for i in range(N):
    H_nuc += -f_L_MHz * I_z[i]

# Nuclear dipolar couplings
# Convert coordinates to meters for dipolar formula
coords_m = coords_A * 1e-10
for i in range(N):
    for j in range(i + 1, N):
        r_vec = coords_m[j] - coords_m[i]
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        # Dipolar coupling constant (convert J to Hz, then to MHz)
        coupling_Hz = (mu0 / (4 * np.pi)) * (gamma_C * hbar) ** 2 / (h * r**3)
        coupling_MHz = coupling_Hz / 1e6
        # Construct I_i · I_j and (I_i · r_hat)(I_j · r_hat)
        I_dot = I_x[i] @ I_x[j] + I_y[i] @ I_y[j] + I_z[i] @ I_z[j]
        Iir = r_hat[0] * I_x[i] + r_hat[1] * I_y[i] + r_hat[2] * I_z[i]
        Ijr = r_hat[0] * I_x[j] + r_hat[1] * I_y[j] + r_hat[2] * I_z[j]
        H_dip_ij = I_dot - 3 * (Iir @ Ijr)
        H_nuc += coupling_MHz * H_dip_ij

# Hyperfine interaction: add each spin's H_hf = S · A_i · I_i.
# In the two-level subspace (|0>, |-1>), we represent S_z by a projector P_e1 = | -1><-1 |
P_e1 = np.array([[0, 0], [0, 1]], dtype=complex)  # projects onto m_S=-1 state
I_elec = np.eye(2, dtype=complex)
# Start with full Hamiltonian in electron⊗nuclear space
dim_total = 2 * dim_nuc
H_total = np.kron(I_elec, H_nuc)  # electron identity ⊗ nuclear H_nuc
# Add hyperfine terms: effectively P_e1 ⊗ [ - (A_xz I_x + A_yz I_y + A_zz I_z) ] for each spin
# (The factor -1 comes from S_z = -1 for the |-1> state).
for i, A in enumerate(A_tensors):
    _, _, Azz, _, Axz, Ayz = (
        A  # (we use only A_xz, A_yz, A_zz for coupling along NV axis)
    )
    H_hf_i = -(Axz * I_x[i] + Ayz * I_y[i] + Azz * I_z[i])
    H_total += np.kron(P_e1, H_hf_i)

# Symmetrize to ensure H is Hermitian
H_total = (H_total + H_total.conj().T) / 2
print("Hamiltonian matrix size:", H_total.shape)


# Diagonalize the total Hamiltonian
E_vals, V = LA.eigh(H_total)  # eigenvalues (MHz) and eigenvectors
# Precompute initial state vectors for each nuclear basis state
# Basis: electron index (0 or 1) × nuclear basis index (0...2^N-1)
# Represent nuclear basis states as bitstrings of length N (0=up, 1=down)
up = np.array([1, 0], dtype=complex)  # |↑⟩ nuclear
dn = np.array([0, 1], dtype=complex)  # |↓⟩ nuclear
initial_states = []
for n_index in range(dim_nuc):
    # Build nuclear basis vector |n⟩:
    state_n = np.array([1], dtype=complex)
    for b in range(N):
        bit = (n_index >> (N - 1 - b)) & 1  # bit of n_index (spin b)
        state_n = np.kron(state_n, (up if bit == 0 else dn))
    # Combine with electron superposition (|0> + |1>)/√2:
    psi0 = np.zeros(dim_total, dtype=complex)
    psi0[0:dim_nuc] = state_n  # electron |0> block
    psi0[dim_nuc:dim_total] = state_n  # electron |-1> block
    psi0 /= np.sqrt(2)
    initial_states.append(psi0)

# Transform all initial states to the eigenbasis for faster propagation
V_inv = V.conj().T
initial_eig = [V_inv.dot(psi0) for psi0 in initial_states]

# Time evolution and echo simulation
taus = np.linspace(0, 200, 400)  # tau from 0 to 60 µs in steps of 0.2 µs
echo_coherence = []  # to store ⟨S_x⟩ coherence (normalized)
for tau in taus:
    # Evolution for time tau/2
    phase = np.exp(-1j * 2 * np.pi * E_vals * (tau / 2))  # phase factors e^{-iE tau/2}
    # Accumulate electron coherence over all nuclear initial states
    coh_sum = 0 + 0j
    for psi0_eig in initial_eig:
        # state after first free evolution
        psi_half_eig = psi0_eig * phase  # apply phase in eigen basis
        psi_half = V.dot(psi_half_eig)  # back to original basis
        # Apply pi-pulse (swap electron components):
        psi_half_swapped = np.zeros_like(psi_half)
        psi_half_swapped[0:dim_nuc] = psi_half[dim_nuc:dim_total]
        psi_half_swapped[dim_nuc:dim_total] = psi_half[0:dim_nuc]
        # Second free evolution for tau/2
        psi_half_swapped_eig = V_inv.dot(psi_half_swapped)
        psi_final_eig = psi_half_swapped_eig * phase
        psi_final = V.dot(psi_final_eig)
        # Compute electron coherence = ⟨0|ρ|1⟩ for this final state
        # (sum over nuclear components of electron-0 amplitude * conj(electron-1 amplitude))
        amp0 = psi_final[0:dim_nuc]  # electron |0> part
        amp1 = psi_final[dim_nuc:dim_total]  # electron |-1> part
        coh = np.vdot(amp1, amp0)  # inner product = sum_i amp0[i]*conj(amp1[i])
        coh_sum += coh
    # Ensemble-averaged coherence:
    coh_avg = coh_sum / dim_nuc
    # For normalized echo signal, divide by initial coherence (at tau=0)
    # At tau=0, coh_avg = 0.5 (because ρ_off-diag = 0.5 for a pure superposition)
    echo_coherence.append(np.real(coh_avg * 2))
    import matplotlib.pyplot as plt


def gaussian_envelope(t, T2):
    return np.exp(-((t / T2) ** 2))


# Insert just before the Gaussian fit and plot
# Apply semiclassical correction from remaining bath spins
def simulate_semiclassical_decoherence(Azz, taus, gamma=175.6, N_trials=1000):
    """Semiclassical echo decay from distant bath spins using Azz components."""
    N = len(Azz)
    coh = []
    for tau in taus * 1e-6:  # convert to seconds
        s = []
        for _ in range(N_trials):
            phase = 0
            for k in range(N):
                Iz = np.random.choice([0.5, -0.5])
                flip = np.random.rand() < 1 - np.exp(-gamma * tau)
                if flip:
                    phase += -2 * np.pi * Azz[k] * Iz * tau  # phase in rad
            s.append(np.cos(phase))
        coh.append(np.mean(s))
    return np.array(coh)


# Prepare semiclassical bath (excluding top 10 indices used in quantum sim)
full_Azz = data["Azz"].values
# remaining_indices = list(set(data.index) - set(indices))
# Azz_rest = full_Azz[remaining_indices]

# Simulate semiclassical bath decoherence
semi_classical_decay = simulate_semiclassical_decoherence(full_Azz, taus)

# Multiply full quantum result with semiclassical decay
hybrid_echo = np.array(echo_coherence) * semi_classical_decay

# Update fit and plot
popt, _ = curve_fit(gaussian_envelope, taus, hybrid_echo, p0=[50])
T2_est = popt[0]
print(f"Hybrid Estimated T2 (quantum + semiclassical): {T2_est:.2f} µs")

plt.figure(figsize=(10, 4))
plt.plot(taus, hybrid_echo, label="Hybrid Echo", color="orange")
plt.xlabel("Echo delay $\\tau$ ($\\mu$s)")
plt.ylabel("Echo coherence (normalized)")
plt.title("Hahn Echo with Quantum + Semiclassical Bath")
plt.grid(True)
for k in [1, 2, 3, 4]:
    plt.text(k * 1e6 / f_L + 0.2, 0.55, f"{k}×$T_L$", color="red", rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# popt, _ = curve_fit(gaussian_envelope, taus, echo_coherence, p0=[50])
# T2_est = popt[0]
# print(f"Estimated T₂ (from Gaussian envelope fit): {T2_est:.2f} µs")

# plt.figure(figsize=(10, 4))
# plt.plot(taus, echo_coherence, color="orange")
# plt.xlabel("Echo delay $\\tau$ ($\\mu$s)")
# plt.ylabel("Echo coherence (normalized)")
# plt.title("Hahn Echo Collapse and Revival (NV center with $^{13}$C bath)")
# plt.grid(True)
# # Mark the Larmor period (~19.6 µs) and multiples
# for k in [1, 2, 3, 4]:
#     # plt.axvline(k * 1e6 / f_L, ls="--", color="red", alpha=0.5)
#     plt.text(k * 1e6 / f_L + 0.2, 0.55, f"{k}×$T_L$", color="red", rotation=90)
# plt.tight_layout()
# plt.show()
