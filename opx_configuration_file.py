import numpy as np


#######################
# AUXILIARY FUNCTIONS #
#######################

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############

qop_ip = "128.104.160.117"
analog_output_delay = 136 #ns
# APD indices, telling you which of the two APDs we are actually using right now
apd_indices = [0, 1]

# Frequencies
NV_IF_freq = 40e6  # in units of Hz
NV2_IF_freq = 45e6
NV_LO_freq = 2.83e9  # in units of Hz

# Pulses lengths
initialization_len = 200  # in ns
meas_len = 100  # in ns
long_meas_len = 5e2  # in ns

# MW parameters
mw_amp_NV = 0.2  # in units of volts
mw_len_NV = 200  # in units of ns

pi_amp_NV = 0.1  # in units of volts
pi_len_NV = 100  # in units of ns

pi_half_amp_NV = pi_amp_NV / 2  # in units of volts
pi_half_len_NV = pi_len_NV  # in units of ns

# Readout parameters
signal_threshold = -500

# Delays
detection_delay = 36 # keep at 36ns minimum
mw_delay = 0

#uwave length. doesn't really matter
uwave_len = 16


green_laser_delay = 0
red_laser_delay = 0
apd_0_delay = 0
apd_1_delay = 0
uwave_delay = 0
delays = [green_laser_delay,apd_0_delay,apd_1_delay,uwave_delay]

common_delay = analog_output_delay + max(delays)

green_laser_total_delay = common_delay - green_laser_delay
red_laser_total_delay = common_delay - red_laser_delay
apd_0_total_delay = common_delay - apd_0_delay
apd_1_total_delay = common_delay - apd_0_delay
uwave_total_delay = common_delay - uwave_delay
NV_total_delay = common_delay + mw_delay
NV2_total_delay = common_delay + mw_delay

config_opx = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": NV_total_delay},  # NV I  
                2: {"offset": 0.0, "delay": NV2_total_delay},  # NV Q
            },
            "digital_outputs": {
                1: {},  # green_laser_do/Laser
                2: {},  #
                3: {}, 
                4: {},   # red laser
            },
            "analog_inputs": {
                1: {"offset": 0},  # APD0
                2: {"offset": 0},  # APD1
            },
        }
    },
    "elements": {
        "NV": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2), "lo_frequency": NV_LO_freq, "mixer": "mixer_NV"},
            "intermediate_frequency": NV_IF_freq,
            "operations": {
                "cw": "const_pulse",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
            },
        },
        "NV2": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2), "lo_frequency": NV_LO_freq, "mixer": "mixer_NV2"},
            "intermediate_frequency": NV2_IF_freq,
            "operations": {
                "cw": "const_pulse",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
            },
        },
        "do_signal_generator": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 3),
                    "delay": uwave_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "uwave_ON": "uwave_ON",    
            },
        },
        "green_laser_do": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1),
                    "delay": green_laser_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON",
            },
        },
        
        "red_laser_do": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 4),
                    "delay": red_laser_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON",
            },
        },
        "APD_0": {
            "singleInput": {"port": ("con1", 1)},  
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 2),
                    "delay": apd_0_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "outputPulseParameters": {
                "signalThreshold": signal_threshold,
                "signalPolarity": "Descending",
                "derivativeThreshold": 1023,
                "derivativePolarity": "Descending",
            },
            "time_of_flight": detection_delay,
            "smearing": 18, #tries to account for length of count pulses being finite. 
        },
        "APD_1": {
            "singleInput": {"port": ("con1", 2)},  
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 2),
                    "delay": apd_1_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "outputPulseParameters": {
                "signalThreshold": signal_threshold,
                "signalPolarity": "Descending",
                "derivativeThreshold": 1023,
                "derivativePolarity": "Descending",
            },
            "time_of_flight": detection_delay,
            "smearing": 18,
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "cw_wf", "Q": "zero_wf"},
        },
        "x180_pulse": {
            "operation": "control",
            "length": pi_len_NV,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "x90_pulse": {
            "operation": "control",
            "length": pi_half_len_NV,
            "waveforms": {"I": "pi_half_wf", "Q": "zero_wf"},
        },
        "laser_ON": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "uwave_ON": {
            "operation": "control",
            "length": uwave_len,
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": meas_len,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_meas_len,
            "digital_marker": "OFF",
            "waveforms": {"single": "zero_wf"},
        },
    },
    "waveforms": {
        "cw_wf": {"type": "constant", "sample": mw_amp_NV},
        "pi_wf": {"type": "constant", "sample": pi_amp_NV},
        "pi_half_wf": {"type": "constant", "sample": pi_half_amp_NV},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
    "mixers": {
        "mixer_NV": [
            {"intermediate_frequency": NV_IF_freq, "lo_frequency": NV_LO_freq, "correction": IQ_imbalance(0.0, 0.0)},
        ],
        "mixer_NV2": [
            {"intermediate_frequency": NV2_IF_freq, "lo_frequency": NV_LO_freq, "correction": IQ_imbalance(0.0, 0.0)},
        ],
    },
}
