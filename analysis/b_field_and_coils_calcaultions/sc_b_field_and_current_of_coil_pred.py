# nv_calib.py
import numpy as np

# --- bring your exact Hamiltonian quartet (crystal frame) ---
# Must return quartet in NV-axes order [ [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1] ]
from sc_b_field_of_coils import exact_f_minus_quartet  # your exact model

# ===== Locked-in calibration (replace only if you re-fit) =====
B0_REF = np.array([-46.275577, -17.165999,  -5.701398], float)
K_REF  = np.column_stack([
    [ +0.803449,  -4.758891, -15.327336],   # dB/dI_ch1 (G/A)
    [ -0.741645,  +4.202657,  -4.635283],   # dB/dI_ch2 (G/A)
]).astype(float)
# Optional: common coil-on offset for THAT old session only
DRIFT_OLD = np.array([ +6.134025, -4.403519, +5.701398 ], float)

# ODMR constants (tune if needed)
D_GHZ = 2.8785
GAMMA = 2.8025  # MHz/G
E_MHZ = 0.0

class NVCalib:
    """
    Minimal calibration wrapper:
      B(I) = B0 + K @ [I1, I2] + (drift if provided)
    """
    def __init__(self, B0=B0_REF, K=K_REF, D_GHz=D_GHZ, gamma_MHz_per_G=GAMMA, E_MHz=E_MHZ):
        self.B0 = np.asarray(B0, float).reshape(3)
        self.K  = np.asarray(K,  float).reshape(3,2)
        self.D  = float(D_GHz)
        self.g  = float(gamma_MHz_per_G)
        self.E  = float(E_MHz)
        self._drift = None  # not used unless set

    # ---------- drift control (optional) ----------
    def set_drift(self, drift_vec_or_None):
        """Set a 3-vector drift (Gauss) that is ADDED whenever you predict."""
        self._drift = None if drift_vec_or_None is None else np.asarray(drift_vec_or_None, float).reshape(3)

    # ---------- core forward maps ----------
    def predict_B(self, I1, I2, include_drift=False):
        """
        Returns:
          B (3,), |B|, B_hat (3,)
        Prints:
          - 'B (G)', '|B| (G)', 'B_hat'
        """
        I = np.array([float(I1), float(I2)], float)
        B = self.B0 + self.K @ I
        if include_drift and (self._drift is not None):
            B = B + self._drift
        Bmag = float(np.linalg.norm(B))
        Bhat = (B / Bmag) if Bmag > 0 else np.zeros(3)
        print("\n=== Field prediction ===")
        print("Inputs: I_ch1 = {:.6f} A, I_ch2 = {:.6f} A".format(I1, I2))
        print("B (G):   ", np.round(B, 6))
        print("|B| (G): ", round(Bmag, 6))
        print("B_hat:   ", np.round(Bhat, 6))
        return B, Bmag, Bhat

    def predict_peaks(self, I1, I2, include_drift=False):
        """
        Returns:
          quartet in NV-axes order (GHz)
        Prints:
          - 'Quartet f_- (GHz) in NV-axes order' per NV
        """
        B, *_ = self.predict_B(I1, I2, include_drift=include_drift)
        f4 = exact_f_minus_quartet(B, D_GHz=self.D, E_MHz=self.E, gamma_e_MHz_per_G=self.g)
        labs = ["[1,1,1]","[-1,1,1]","[1,-1,1]","[1,1,-1]"]
        print("\nQuartet f_- (GHz) in NV-axes order:")
        for lab, f in zip(labs, f4):
            print(f" {lab:<10} {f:.9f}")
        return f4

    # ---------- inverse problems on the linear B-map ----------
    def currents_for_deltaB(self, dB_target_G, clip_A=None):
        """
        Solve LS:  K @ [I1,I2] ≈ dB_target
        Returns dict: Iy_A, Iz_A, dB_achieved_G, dB_resid_G
        Prints:
          - requested ΔB, achieved ΔB, residual
        """
        dB = np.asarray(dB_target_G, float).reshape(3)
        I = np.linalg.lstsq(self.K, dB, rcond=None)[0]  # (2,)
        if clip_A is not None:
            I = np.clip(I, -float(clip_A), +float(clip_A))
        dB_ach = self.K @ I
        resid  = dB_ach - dB
        print("\n=== Currents for target ΔB (linear LS) ===")
        print("ΔB_target (G):   ", np.round(dB, 6))
        print("ΔB_achieved (G): ", np.round(dB_ach, 6))
        print("ΔB_residual (G): ", np.round(resid, 6))
        print("I_ch1 (A), I_ch2 (A):", float(I[0]), float(I[1]))
        return {"Iy_A": float(I[0]), "Iz_A": float(I[1]),
                "dB_achieved_G": dB_ach, "dB_resid_G": resid}

    def currents_to_scale_Bmag(self, scale=1.10, clip_A=None):
        """
        Keep direction, scale |B| by 'scale' around B0.
        Returns dict like currents_for_deltaB.
        Prints:
          - target |B|, ΔB target, achieved, residual, currents
        """
        Bmag = float(np.linalg.norm(self.B0))
        if Bmag <= 0:
            raise ValueError("B0 magnitude is zero; cannot scale.")
        Bhat = self.B0 / Bmag
        B1   = Bhat * (Bmag * float(scale))
        dB   = B1 - self.B0
        print("\n=== Currents to scale |B| ===")
        print(f"Target scale: {scale:.4f}   |B0|→|B1|: {Bmag:.3f}→{np.linalg.norm(B1):.3f} G")
        return self.currents_for_deltaB(dB, clip_A=clip_A)

    # def currents_for_target_B(self, B_target_G, clip_A=None):
    #     """
    #     Hit an explicit crystal-frame B_target.
    #     Returns dict like currents_for_deltaB.
    #     Prints:
    #       - target B, ΔB target, achieved, residual, currents
    #     """
    #     B_tgt = np.asarray(B_target_G, float).reshape(3)
    #     dB = B_tgt - self.B0
    #     print("\n=== Currents for target B (crystal frame) ===")
    #     print("B_target (G):    ", np.round(B_tgt, 6))
    #     return self.currents_for_deltaB(dB, clip_A=clip_A)
      
      # add to your NVCalib class

    def _project_into_spanK(self, dB):
        Kpinv = np.linalg.pinv(self.K)        # 2x3
        dB_proj = self.K @ (Kpinv @ dB)       # projection of dB onto Col(K)
        resid   = dB - dB_proj
        # reachability score: cos(angle) between dB and its projection
        num = float(np.dot(dB, dB_proj))
        den = float(np.linalg.norm(dB) * np.linalg.norm(dB_proj)) + 1e-12
        reach = np.clip(num/den, -1.0, 1.0) if den > 0 else 0.0
        return dB_proj, resid, reach

    def currents_for_target_B(self, B_target_G, clip_A=None, ridge=0.0):
        """
        Hit an explicit crystal-frame B_target as well as possible within span(K).
        If the target is not reachable, we automatically project it into span(K)
        and report the reachability score in [0,1]. Optionally apply 'ridge' (A^-1)
        and clip currents to +/- clip_A (A).
        """
        B_tgt = np.asarray(B_target_G, float).reshape(3)
        dB_req = B_tgt - self.B0
        dB_proj, dB_orth, reach = self._project_into_spanK(dB_req)

        # LS solve (with optional ridge): (K^T K + λI) I = K^T dB_proj
        KtK = self.K.T @ self.K
        if ridge > 0:
            KtK = KtK + float(ridge) * np.eye(2)
        I = np.linalg.solve(KtK, self.K.T @ dB_proj)

        if clip_A is not None:
            I = np.clip(I, -float(clip_A), +float(clip_A))

        dB_ach = self.K @ I
        resid  = dB_ach - dB_req
        B_ach  = self.B0 + dB_ach
        Bmag   = float(np.linalg.norm(B_ach))

        print("\n=== Currents for target B (with reachability diagnostics) ===")
        print("B_target (G):        ", np.round(B_tgt, 6))
        print("ΔB_requested (G):    ", np.round(dB_req, 6))
        print("ΔB_projected (G):    ", np.round(dB_proj, 6), "  (in span(K))")
        print("ΔB_unreachable (G):  ", np.round(dB_orth, 6), "  (orthogonal to span(K))")
        print(f"Reachability score:   {reach:.3f}  (1=fully reachable, 0=orthogonal)")
        print("Currents (A):         I_ch1={:.6f}, I_ch2={:.6f}".format(float(I[0]), float(I[1])))
        print("ΔB_achieved (G):     ", np.round(dB_ach, 6))
        print("Residual to target:  ", np.round(resid, 6))
        print("B_achieved (G):      ", np.round(B_ach, 6), f"  |B|={Bmag:.6f} G")
        return {
            "Iy_A": float(I[0]), "Iz_A": float(I[1]),
            "reachability": float(reach),
            "dB_requested_G": dB_req, "dB_projected_G": dB_proj,
            "dB_unreachable_G": dB_orth,
            "dB_achieved_G": dB_ach, "dB_residual_G": resid,
            "B_achieved_G": B_ach, "B_achieved_mag_G": Bmag,
        }

      
    def peaks_at_currents(self, Iy, Iz, exact_quartet_fn=None):
        """
        Return (B, |B|, f_exact) for given currents.
        exact_quartet_fn: callable(B_crys_G, D_GHz, E_MHz, gamma_e_MHz_per_G) -> (4,)
        """
        dB = self.K @ np.array([Iy, Iz], float)
        B  = self.B0 + dB
        Bmag = float(np.linalg.norm(B))
        f_exact = None
        if exact_quartet_fn is not None:
            f_exact = exact_quartet_fn(B, D_GHz=self.D, E_MHz=self.E, gamma_e_MHz_per_G=self.g)
        return B, Bmag, f_exact

    def print_plan_with_peaks(self, plan, exact_quartet_fn=None, nv_labels=None):
        """Pretty-print the plan, |B|, and (optional) exact quartet."""
        if nv_labels is None:
            nv_labels = ["[1,1,1]","[-1,1,1]","[1,-1,1]","[1,1,-1]"]

        Iy, Iz = plan["Iy_A"], plan["Iz_A"]
        B, Bmag, f_exact = self.peaks_at_currents(Iy, Iz, exact_quartet_fn)

        print("\n=== Field prediction ===")
        print(f"Inputs: I_ch1 = {Iy:.6f} A, I_ch2 = {Iz:.6f} A")
        print("B (G):   ", np.round(B, 6))
        print(f"|B| (G):  {Bmag:.6f}")

        if f_exact is not None:
            print("\nQuartet f_- (GHz) in NV-axes order:")
            for lab, v in zip(nv_labels, f_exact):
                print(f" {lab:<10} {v:.9f}")
if __name__ == "__main__":
  cal = NVCalib()                 # uses your B0 & K above
  # cal.set_drift(DRIFT_OLD)      # only if you want to correct that old session

  # 1) Predict peaks for a shot
  f = cal.predict_peaks(I1=-3.0, I2=-2.0)   # prints B, |B|, B_hat, and quartet

  # 2) Scale |B| by +10% along current direction
  # plan = cal.currents_to_scale_Bmag(scale=1.10)  # prints plan; plan['Iy_A'], plan['Iz_A']

  # 3) Aim for an explicit target B
#   planB = cal.currents_for_target_B([-120.0, -18.0, -10.0])

  # 4) Solve currents for a desired ΔB directly
#   planD = cal.currents_for_deltaB([-0.56, +3.0, -18.3])

  # # 5) If you need the exact quartet at the planned B:
#   B_planned, *_ = cal.predict_B(planB["Iy_A"], planB["Iz_A"])
#   f_exact = exact_f_minus_quartet(B_planned, D_GHz=cal.D, E_MHz=cal.E, gamma_e_MHz_per_G=cal.g)
  
#   planB = cal.currents_for_target_B([-120.0, -18.0, -10.0], clip_A=10.0, ridge=0.0)
  # Inspect:
  # - planB["reachability"]  (likely very low because of huge -x request)
  # - planB["dB_unreachable_G"]  (the part K cannot make)
  # - planB["B_achieved_G"], then if you want exact peaks at that B:
#   B_ach = planB["B_achieved_G"]
#   f_exact = exact_f_minus_quartet(B_ach, D_GHz=cal.D, E_MHz=cal.E, gamma_e_MHz_per_G=cal.g)

  # 2) Get currents for a (possibly unreachable) target B with diagnostics
#   planB = cal.currents_for_target_B([-45.0, -35.0, -20.0], clip_A=10.0, ridge=0.0)

  # 3) Print total |B| and exact quartet at the achieved B
#   cal.print_plan_with_peaks(planB, exact_quartet_fn=exact_f_minus_quartet)

  # If you just want the raw values programmatically:
#   B_ach, Bmag_ach, f_exact = cal.peaks_at_currents(planB["Iy_A"], planB["Iz_A"], exact_quartet_fn=exact_f_minus_quartet)
#   print("\n(B_achieved, |B|, first f_-):", np.round(B_ach,6), f"{Bmag_ach:.6f}", f"{f_exact[0]:.9f} GHz")