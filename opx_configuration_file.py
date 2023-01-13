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
# apd_indices = [0,1]

# Frequencies
NV_IF_freq = 40e6  # in units of Hz
NV2_IF_freq = 45e6
NV_LO_freq = 2.83e9  # in units of Hz

# Pulses lengths
initialization_len = 200  # in ns
meas_len = 100  # in ns
long_meas_len = 100  # in ns

# MW parameters
mw_amp_NV = .5  # in units of volts
mw_len_NV = 200  # in units of ns

aom_amp = 0.5

pi_amp_NV = 0.1  # in units of volts
pi_len_NV = 100  # in units of ns

pi_half_amp_NV = pi_amp_NV / 2  # in units of volts
pi_half_len_NV = pi_len_NV  # in units of ns

# Readout parameters
signal_threshold = -200

# Delays
detection_delay = 36 # keep at 36ns minimum
mw_delay = 0

#uwave length. doesn't really matter
uwave_len = 100


green_laser_delay = 0
red_laser_delay = 0
apd_0_delay = 0
apd_1_delay = 0
uwave_delay = 0
aod_delay = 0
yellow_aom_delay = 0
tsg4104_I_delay = 0
tsg4104_Q_delay = 0
delays = [green_laser_delay, red_laser_delay, apd_0_delay,apd_1_delay, uwave_delay, aod_delay,
          yellow_aom_delay, tsg4104_I_delay, tsg4104_Q_delay]

min_delay = 150 #we use 100 with the pulse streamer. doesn't matter. just wanted it higher than 136 analog delay

common_delay = max(delays) + min_delay 

green_laser_total_delay = common_delay - green_laser_delay
red_laser_total_delay = common_delay - red_laser_delay
apd_0_total_delay = common_delay - apd_0_delay
apd_1_total_delay = common_delay - apd_0_delay
uwave_total_delay = common_delay - uwave_delay
NV_total_delay = common_delay - mw_delay
NV2_total_delay = common_delay - mw_delay
AOD_total_delay = common_delay - aod_delay
yellow_AOM_total_delay = common_delay - yellow_aom_delay
tsg4104_I_total_delay = common_delay - tsg4104_I_delay
tsg4104_Q_total_delay = common_delay - tsg4104_Q_delay 


config_opx = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": NV_total_delay}, # will be I for sig gen 
                2: {"offset": 0.0, "delay": NV_total_delay}, # will be Q for sig gen
                3: {"offset": 0.0, "delay": AOD_total_delay}, #AOD_1X
                4: {"offset": 0.0, "delay": AOD_total_delay}, #AOD_1Y
                5: {"offset": 0.0, "delay": yellow_AOM_total_delay}, #yellow AOM
                6: {"offset": 0.0, "delay": tsg4104_I_total_delay}, #I for tsg4104 signal generator
                7: {"offset": 0.0, "delay": tsg4104_Q_total_delay}, #Q for tsg4104 signal generator
            },
            "digital_outputs": {
                1: {},  # 
                2: {},  # apd 1 virtual gate
                3: {},  # apd 2 virtual gate
                5: {},  # clock
                7: {},  # tsg4104 sig gen switch
                8: {},  # cobolt 638
                9: {},  # cobolt 515 
            },
            "analog_inputs": {
                1: {"offset": 0},  # APD0
                2: {"offset": 0},  # APD1
            },
        }
    },
    "elements": {
        "AOD_1X": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": NV_IF_freq,
            "operations": {
                "cw": "const_freq_out",
            },
        },
        "AOD_1Y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": NV_IF_freq,
            "operations": {
                "cw": "const_freq_out",
            },
        },
        "sig_gen_TEKT_tsg4104a_I": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 0.0,
            "operations": {
                "cw": "const_freq_out",
            },
        },
        "sig_gen_TEKT_tsg4104a_Q": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0.0,
            "operations": {
                "cw": "const_freq_out",
            },
        },
        "laserglow_589": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": 0,
            "operations": {
                "laser_ON_ANALOG": "laser_ON_ANALOG",
                "cw": "const_freq_out",
            },
        },
        "sig_gen_TEKT_tsg4104a": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 7),
                    "delay": uwave_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "uwave_ON": "uwave_ON",
                "uwave_OFF": "uwave_OFF",
                "constant_HIGH": "constant_HIGH",
            },
        },
        "cobolt_515": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 9),
                    "delay": green_laser_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON_DIGITAL": "laser_ON_DIGITAL",
                "laser_OFF_DIGITAL": "laser_OFF_DIGITAL",
                "constant_HIGH": "constant_HIGH",
            },
        },
        "cobolt_638": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 8),
                    "delay": red_laser_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON_DIGITAL": "laser_ON_DIGITAL",
                "laser_OFF_DIGITAL": "laser_OFF_DIGITAL",
                "constant_HIGH": "constant_HIGH",
            },
        },
        "do_sample_clock": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 5),
                    "delay": common_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "clock_pulse": "clock_pulse",
                "zero_clock_pulse": "zero_clock_pulse",
            },
        },
        "do_apd_0_gate": {
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
                "signalPolarity": "Below",
                "derivativeThreshold": 1800,
                "derivativePolarity": "Below",
            },
            "time_of_flight": detection_delay,
            "smearing": 15, #tries to account for length of count pulses being finite. 
        },
        "do_apd_1_gate": {
            "singleInput": {"port": ("con1", 2)},  
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 3),
                    "delay": apd_1_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 2)},
            "outputPulseParameters": {
                "signalThreshold": signal_threshold,
                "signalPolarity": "Below",
                "derivativeThreshold": 1800,
                "derivativePolarity": "Below",
            },
            "time_of_flight": detection_delay,
            "smearing": 15,
        },
    },
    "pulses": {
        "const_freq_out": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"single": "cw_wf"},
        },
        "laser_ON_ANALOG": {
            "operation": "control",
            "length": initialization_len,
            "waveforms": {"single": "cw_wf"},
        },
        "laser_ON_DIGITAL": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "laser_OFF_DIGITAL": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "OFF",
        },
        "constant_HIGH": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "clock_pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
        },
        "zero_clock_pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "OFF",
        },
        "uwave_ON": {
            "operation": "control",
            "length": uwave_len,
            "digital_marker": "ON",
        },
        "uwave_OFF": {
            "operation": "control",
            "length": uwave_len,
            "digital_marker": "OFF",
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
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
    },
    "waveforms": {
        "cw_wf": {"type": "constant", "sample": mw_amp_NV},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
}
