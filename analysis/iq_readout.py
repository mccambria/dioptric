import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n_shots = 1000  # Number of repetitions per state
sigma = 0.5  # Noise standard deviation
iq0 = (1.0, 0.0)  # Ideal IQ point for |0⟩
iq1 = (-1.0, 0.0)  # Ideal IQ point for |1⟩

# Generate noisy samples around each IQ point
I0 = np.random.normal(loc=iq0[0], scale=sigma, size=n_shots)
Q0 = np.random.normal(loc=iq0[1], scale=sigma, size=n_shots)

I1 = np.random.normal(loc=iq1[0], scale=sigma, size=n_shots)
Q1 = np.random.normal(loc=iq1[1], scale=sigma, size=n_shots)

# Plot IQ blobs
plt.figure(figsize=(6, 5))
plt.scatter(I0, Q0, color="blue", label="|0⟩", alpha=0.5)
plt.scatter(I1, Q1, color="red", label="|1⟩", alpha=0.5)
plt.axvline(x=0, color="gray", linestyle="--", label="I threshold")
plt.xlabel("I")
plt.ylabel("Q")
plt.title("Simulated Qubit Readout IQ Blobs")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
