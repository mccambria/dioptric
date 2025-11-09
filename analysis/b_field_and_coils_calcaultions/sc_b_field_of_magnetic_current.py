import numpy as np

class NVCalib:
    def __init__(self, B0, K):
        self.B0 = np.asarray(B0, float).reshape(3)
        self.K  = np.asarray(K,  float).reshape(3,2)

    # ---------- 1) Minimal-norm currents to scale |B| by a factor 'scale' ----------
    def min_norm_currents_to_scale_Bmag(self, scale):
        """
        Minimal-norm I that increases |B| along current B̂ by the desired amount.
        (Small-signal exact; for large changes we add a 1D scalar refine.)
        Returns dict with Iy, Iz, achieved B, and |B|.
        """
        B0   = self.B0
        K    = self.K
        Bmag = float(np.linalg.norm(B0))
        Bhat = B0 / Bmag

        # Desired parallel change in |B|:
        dB_parallel = (float(scale) - 1.0) * Bmag  # Gauss

        # Minimal-norm I satisfying  Bhat·(K I) = dB_parallel  is:
        # I* = dB_parallel * (K^T Bhat) / ||K^T Bhat||^2
        v = K.T @ Bhat                      # shape (2,)
        denom = float(v @ v)
        if denom <= 0:
            raise RuntimeError("K^T Bhat is zero; coils cannot change |B| along current direction.")
        I0 = (dB_parallel / denom) * v      # minimal-norm solution

        # Optional 1D refine (exact magnitude hit): scale I0 by alpha so that |B0 + K (alpha I0)| = scale*|B0|
        # This is a scalar root; do 5-8 Newton steps which is cheap.
        target = scale * Bmag
        I = I0.copy()
        for _ in range(8):
            B = B0 + K @ I
            Bm = float(np.linalg.norm(B))
            if Bm == 0:
                break
            # d|B|/d(alpha) = B_hat(B) · (K I0)
            Bh = B / Bm
            dBm_dalpha = float(Bh @ (K @ I0))
            # Newton step on g(alpha) = |B| - target
            g = Bm - target
            if abs(dBm_dalpha) < 1e-12:
                break
            I -= (g / dBm_dalpha) * I0
            if abs(g) < 1e-6:  # ~1 µG tolerance on |B|
                break

        B1 = B0 + K @ I
        return {
            "Iy_A": float(I[0]), "Iz_A": float(I[1]),
            "B1_G": B1, "B1_mag_G": float(np.linalg.norm(B1)),
            "B0_mag_G": Bmag,
            "scale_target": float(scale)
        }

    # ---------- 2a) Best direction per amp (no budget yet, just the direction) ----------
    def steepest_increase_direction(self):
        """
        Unit-norm current direction that maximizes first-order increase of |B|.
        It's along K^T B̂.
        """
        Bhat = self.B0 / np.linalg.norm(self.B0)
        g = self.K.T @ Bhat
        ng = np.linalg.norm(g)
        if ng == 0:
            raise RuntimeError("K^T Bhat is zero; coils cannot change |B| along current direction.")
        return g / ng  # unit vector in (Iy, Iz)

    # ---------- 2b) Given a current budget, pick currents that maximize |B| ----------
    def maximize_B_with_budget(self, I_budget, n_theta=360):
        """
        Solve argmax_{||I||<=I_budget} |B0 + K I| by scanning angle (fast, 2D).
        Returns optimal (Iy, Iz), B, |B|.
        """
        I_budget = float(I_budget)
        best = None
        for th in np.linspace(0, 2*np.pi, n_theta, endpoint=False):
            I = I_budget * np.array([np.cos(th), np.sin(th)], float)
            B = self.B0 + self.K @ I
            val = float(np.linalg.norm(B))
            if (best is None) or (val > best[0]):
                best = (val, I, B)
        return {"Iy_A": float(best[1][0]), "Iz_A": float(best[1][1]),
                "B_G": best[2], "B_mag_G": best[0]}

    # ---------- Convenience: print a quick summary ----------
    def print_summary(self, tag, Iy, Iz):
        B = self.B0 + self.K @ np.array([Iy, Iz], float)
        print(f"\n[{tag}]")
        print(f"I_ch1={Iy:.6f} A, I_ch2={Iz:.6f} A")
        print("B (G):   ", np.round(B, 6))
        print(f"|B| (G):  {np.linalg.norm(B):.6f}")
# ------------- Example -------------
if __name__ == "__main__":
    # Quartets (GHz), internal order doesn't matter for the solver you’re calling
    # Your numbers:
    B0 = [-46.275577, -17.165999, -5.701398]
    K  = np.column_stack([
        [ +0.803449,  -4.758891, -15.327336],   # dB/dI_ch1 (G/A)
        [ -0.741645,  +4.202657,  -4.635283],   # dB/dI_ch2 (G/A)
    ])

    cal = NVCalib(B0, K)

    # (A) Minimal-norm currents to get, say, +20% in |B|
    ans = cal.min_norm_currents_to_scale_Bmag(scale=2.20)
    cal.print_summary("Min-norm to +20% |B|", ans["Iy_A"], ans["Iz_A"])

    # (B) Best direction per amp (infinitesimal step)
    idir = cal.steepest_increase_direction()   # unit vector (Iy, Iz)
    print("\nSteepest-increase unit direction (Iy, Iz):", np.round(idir, 6))

    # (C) With a current budget (e.g., ||I|| ≤ 2 A), find the best (Iy,Iz)
    opt = cal.maximize_B_with_budget(I_budget=4.0, n_theta=720)
    cal.print_summary("Maximize |B| under ||I||<=2 A", opt["Iy_A"], opt["Iz_A"])
