import matplotlib.pyplot as plt
import numpy as np


# ------------ Plant (first-order) ------------
class FirstOrderPlant:
    def __init__(self, K=1.0, tau=10.0, y0=0.0):
        self.K = K
        self.tau = tau
        self.y = y0

    def step(self, u, dt):
        # dy/dt = (-y + K*u)/tau
        dy = (-self.y + self.K * u) / self.tau
        self.y += dy * dt
        return self.y


# ------------ PID Controller (discrete) ------------
class PID:
    def __init__(self, Kp, Ki, Kd, dt, umin=None, umax=None, d_filter_hz=5.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.umin = umin
        self.umax = umax

        # State
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_deriv = 0.0  # filtered derivative

        # Derivative low-pass filter
        # y_f' = alpha * y' + (1-alpha) * y_f'; alpha = 1 / (1 + 2*pi*dt*f_c)
        wc = 2.0 * np.pi * d_filter_hz
        self.alpha = 1.0 / (1.0 + wc * dt)

    def update(self, setpoint, measurement):
        error = setpoint - measurement

        # Proportional
        P = self.Kp * error

        # Integral (anti-windup with clamping after computing u)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative (on measurement or error? here on error)
        deriv_raw = (error - self.prev_error) / self.dt
        D_raw = self.Kd * deriv_raw
        # Filter derivative
        D = self.alpha * D_raw + (1 - self.alpha) * self.prev_deriv

        # Control signal
        u = P + I + D

        # Anti-windup clamp
        if self.umin is not None:
            u = max(self.umin, u)
        if self.umax is not None:
            u = min(self.umax, u)

        # If clamped, don't integrate (simple back-calculation alternative)
        if (self.umax is not None and u >= self.umax and error > 0) or (
            self.umin is not None and u <= self.umin and error < 0
        ):
            # Rollback last integral step
            self.integral -= error * self.dt
            I = self.Ki * self.integral
            u = P + I + D
            if self.umin is not None:
                u = max(self.umin, u)
            if self.umax is not None:
                u = min(self.umax, u)

        # Save state
        self.prev_error = error
        self.prev_deriv = D

        return u, P, I, D, error


# ------------ Utility metrics ------------
def settling_time(t, y, band=0.02):
    """
    Time to stay within ±band of final value.
    """
    y_final = y[-1]
    lower, upper = y_final - band, y_final + band
    idx = len(y) - 1
    # Find last index that is outside band
    for i in range(len(y) - 1, -1, -1):
        if not (lower <= y[i] <= upper):
            idx = i
            break
    if idx == len(y) - 1:
        return 0.0
    return t[idx + 1]


def overshoot(y, setpoint):
    ymax = np.max(y)
    return max(0.0, ymax - setpoint)


# ------------ Main simulation ------------
if __name__ == "__main__":
    # Simulation params
    dt = 0.1
    T = 60.0  # total time (s)
    t = np.arange(0, T, dt)

    # Setpoint profile
    sp = np.zeros_like(t)
    sp[t >= 2.0] = 1.0  # step at 2 s

    # Plant
    plant = FirstOrderPlant(K=1.0, tau=3.0, y0=0.0)

    # PID gains (try changing these)
    pid = PID(Kp=2.0, Ki=1.0, Kd=0.2, dt=dt, umin=0.0, umax=2.0, d_filter_hz=5.0)

    # Noise level
    noise_std = 0.002  # e.g., 0.002 for temperature-like noise

    # Storage
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    P_term = np.zeros_like(t)
    I_term = np.zeros_like(t)
    D_term = np.zeros_like(t)
    err = np.zeros_like(t)

    # Sim loop
    for k in range(len(t)):
        meas = plant.y + np.random.randn() * noise_std
        u[k], P_term[k], I_term[k], D_term[k], err[k] = pid.update(sp[k], meas)
        y[k] = plant.step(u[k], dt)

    # Metrics
    st = settling_time(t, y, band=0.02)  # band in same units as setpoint
    ov = overshoot(y, sp[-1])

    print(f"Settling time ~ {st:.3f} s")
    print(f"Overshoot     ~ {ov:.3f}")

    # ------------ Plots ------------
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(t, sp, "k--", label="Setpoint")
    axs[0].plot(t, y, label="Output (y)")
    axs[0].set_ylabel("Output")
    axs[0].legend()
    axs[0].set_title("PID Step Response")

    axs[1].plot(t, u, label="u (control)")
    axs[1].set_ylabel("Control")
    axs[1].legend()

    axs[2].plot(t, P_term, label="P")
    axs[2].plot(t, I_term, label="I")
    axs[2].plot(t, D_term, label="D")
    axs[2].set_ylabel("PID Terms")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
