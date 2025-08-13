# -*- coding: utf-8 -*-
"""
Rabi sequence: two APD gates per period (signal, then reference).
Keep the total period constant across all tau by padding with (max_tau - tau).

 Updated by @Saroj Chand on Aug 6, 2025 (hardened).
"""

import matplotlib.pyplot as plt

# If you placed seq_utils in a different package, adjust this import:
from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils

# Optional helpers to auto-pick a valid readout laser from your config
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def get_seq(args):
    """
    Args (list/tuple):
        0: tau_ns        (int)
        1: pol_ns        (int)
        2: readout_ns    (int)
        3: max_tau_ns    (int)  # must be >= tau_ns
        4: uwave_ind     (int)
        5: ro_laser      (str)  # physical laser name (must exist in config)
        6: ro_power      (float|None)  # analog volts if analog modulation; None if digital
    Returns:
        (Sequence, OutputState, [period_ns])
    """
    tau, pol, readout, max_tau, uwave_ind, ro_laser, ro_power = args
    return seq_utils.build_rabi_like_sequence(
        tau_ns=tau,
        pol_ns=pol,
        readout_ns=readout,
        max_tau_ns=max_tau,
        uwave_ind=uwave_ind,
        readout_laser=ro_laser,
        readout_power=ro_power,
    )


def preview(
    tau_ns: int = 200,
    pol_ns: int = 1000,
    readout_ns: int = 300,
    max_tau_ns: int = 2000,
    uwave_ind: int = 0,
    ro_laser: str | None = None,
    ro_power: float | None = None,
):
    """
    Build and plot a single Rabi sequence for visual confirmation.
    - ro_laser defaults to your configured widefield charge-readout laser.
    """
    if ro_laser is None:
        # Choose a valid physical laser name from config to avoid KeyErrors
        ro_laser = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)

    seq, final, (period_ns,) = get_seq(
        [tau_ns, pol_ns, readout_ns, max_tau_ns, uwave_ind, ro_laser, ro_power]
    )
    print(f"[RABI PREVIEW] period = {period_ns} ns  (tau = {tau_ns} ns)")
    # Pulse Streamer client has a built-in plotter for Sequence:
    seq.plot()  # plots digital + analog channels in subplots
    plt.show()
    return seq, period_ns


if __name__ == "__main__":
    # Tweak these to taste, then run this file directly to preview
    # NOTE: If your readout laser is digitally modulated, keep ro_power=None
    #       If analog, set a voltage (e.g. 0.6).
    preview(
        tau_ns=200,
        pol_ns=1000,
        readout_ns=300,
        max_tau_ns=2000,
        uwave_ind=0,
        ro_laser=None,  # auto-pick from config
        ro_power=None,
    )

# if __name__ == "__main__":

# def get_seq(pulse_streamer, config, args):
#     """
#     args layout (must match confocal_utils.get_base_seq_args):
#         0: tau                 (ns, int)
#         1: polarization_time   (ns, int)
#         2: readout             (ns, int)
#         3: max_tau             (ns, int)  -- used to pad the first arm so the period is constant
#         4: state               (int -> tool_belt.States enum) selects microwave source index
#         5: laser_name          (str)      physical laser name for polarization/readout
#         6: laser_power         (float or None) laser power setpoint (tool_belt handles None)
#     """
#     # Cast the first four durations to int64
#     tau = np.int64(args[0])
#     pol_time = np.int64(args[1])
#     readout = np.int64(args[2])
#     max_tau = np.int64(args[3])

#     # MW source selector
#     state = States(args[4])
#     sig_gen_name = config["Servers"][f"sig_gen_{state.name}"]

#     # Laser info
#     laser_name = args[5]
#     laser_power = args[6]

#     # Wiring (digital channels)
#     wiring = config["Wiring"]["PulseGen"]
#     do_apd_gate = wiring["do_apd_gate"]
#     do_sig_gate = wiring[f"do_{sig_gen_name}_gate"]
#     do_sample_clk = wiring["do_sample_clock"]

#     # Device-specific delays
#     laser_delay = config["Optics"][laser_name]["delay"]
#     uwave_delay = config["Microwaves"][sig_gen_name]["delay"]

#     # Common timing buffers
#     short_buffer = np.int64(10)  # avoid 0-length glitches
#     uwave_buffer = np.int64(config["CommonDurations"]["uwave_buffer"])
#     common_delay = np.int64(max(laser_delay, uwave_delay)) + short_buffer

#     # Keep the laser on exactly as needed for the longer of polarization / first readout
#     readout_pol_max = np.int64(max(readout, pol_time)) + short_buffer
#     final_readout_buffer = np.int64(500)

#     seq = Sequence()

#     # -------------------- APD gate (signal, then reference) --------------------
#     apd_train = [
#         (common_delay, Digital.LOW),
#         (pol_time, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout, Digital.HIGH),  # Signal gate
#         (readout_pol_max - readout, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout, Digital.HIGH),  # Reference gate
#         (final_readout_buffer + short_buffer, Digital.LOW),
#     ]
#     seq.setDigital(do_apd_gate, apd_train)
#     period = np.int64(sum(d for d, _ in apd_train))

#     # -------------------- Laser (pol + both readouts) --------------------
#     laser_train = [
#         (common_delay - np.int64(laser_delay), Digital.LOW),
#         (pol_time, Digital.HIGH),  # Polarization
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),  # Wait through MW first arm (padded to max_tau)
#         (uwave_buffer, Digital.LOW),
#         (readout_pol_max, Digital.HIGH),  # First readout (covers readout duration)
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),  # Second arm (again max_tau for constant period)
#         (uwave_buffer, Digital.LOW),
#         (readout + final_readout_buffer, Digital.HIGH),  # Second readout & settle
#         (short_buffer, Digital.LOW),
#         (np.int64(laser_delay), Digital.LOW),
#     ]
#     tb.process_laser_seq(
#         pulse_streamer, seq, config, laser_name, laser_power, laser_train
#     )

#     # -------------------- Microwave gate (only tau wide in the first arm) --------------------
#     mw_train = [
#         (common_delay - np.int64(uwave_delay), Digital.LOW),
#         (pol_time, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau - tau, Digital.LOW),  # pad so the arm is fixed-length
#         (tau, Digital.HIGH),  # actual Rabi pulse
#         (uwave_buffer, Digital.LOW),
#         (readout_pol_max, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),  # second arm: no pulse, just wait (same length)
#         (uwave_buffer, Digital.LOW),
#         (readout + final_readout_buffer, Digital.LOW),
#         (short_buffer, Digital.LOW),
#         (np.int64(uwave_delay), Digital.LOW),
#     ]
#     seq.setDigital(do_sig_gate, mw_train)

#     final_digital = [do_sample_clk]  # keep sample clock line low at the end
#     final = OutputState(final_digital, 0.0, 0.0)
#     return seq, final, [period]


# if __name__ == "__main__":
#     config = tb.get_config_dict()
#     tb.set_delays_to_zero(config)  # helper for plotting sanity
#     args = [100, 1000, 300, 300, 0, "laserglow_532", None]
#     s, _, _ = get_seq(None, config, args)
#     s.plot()
