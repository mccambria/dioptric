
# Single QUA script generated at 2023-07-10 15:02:03.125744
# QUA library version: 1.1.3

from qm.qua import *

with program() as prog:
    v1 = declare(fixed, value=0.4)
    v2 = declare(fixed, value=0.4)
    v3 = declare(fixed, value=0.4)
    update_frequency("ao1", 110000000.0, "Hz", False)
    update_frequency("ao2", 110000000.0, "Hz", False)
    update_frequency("ao3", 110000000.0, "Hz", False)
    with infinite_loop_():
        play("cw"*amp(v1), "ao1", duration=250)
        play("cw"*amp(v2), "ao2", duration=250)
        play("cw"*amp(v3), "ao3", duration=250)


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "3": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "4": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "5": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "6": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "7": {
                    "offset": 0.0,
                    "delay": 150,
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                },
            },
            "digital_outputs": {
                "1": {},
                "2": {},
                "3": {},
                "4": {},
                "5": {},
                "6": {},
                "7": {},
                "8": {},
                "9": {},
                "10": {},
            },
            "analog_inputs": {
                "1": {
                    "offset": 0,
                },
                "2": {
                    "offset": 0,
                },
            },
        },
    },
    "elements": {
        "do1": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do2": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do3": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do4": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do5": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 5),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do6": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 6),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do7": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 7),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do8": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 8),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do9": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 9),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do10": {
            "digitalInputs": {
                "chan": {
                    "port": ('con1', 10),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "ao1": {
            "singleInput": {
                "port": ('con1', 1),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao2": {
            "singleInput": {
                "port": ('con1', 2),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao3": {
            "singleInput": {
                "port": ('con1', 3),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao4": {
            "singleInput": {
                "port": ('con1', 4),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao5": {
            "singleInput": {
                "port": ('con1', 5),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao6": {
            "singleInput": {
                "port": ('con1', 6),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao7": {
            "singleInput": {
                "port": ('con1', 7),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao8": {
            "singleInput": {
                "port": ('con1', 8),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao9": {
            "singleInput": {
                "port": ('con1', 9),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "ao10": {
            "singleInput": {
                "port": ('con1', 10),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "laserglow_589_x": {
            "singleInput": {
                "port": ('con1', 1),
            },
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
        },
        "sig_gen_TEKT_tsg4104a": {
            "digitalInputs": {
                "marker": {
                    "port": ('con1', 7),
                    "delay": 150,
                    "buffer": 0,
                },
            },
            "operations": {
                "uwave_on": "do_on",
                "uwave_off": "do_off",
            },
        },
        "cobolt_515": {
            "digitalInputs": {
                "marker": {
                    "port": ('con1', 9),
                    "delay": 150,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_on": "do_on",
                "laser_off": "do_off",
            },
        },
        "do_sample_clock": {
            "digitalInputs": {
                "marker": {
                    "port": ('con1', 5),
                    "delay": 150,
                    "buffer": 0,
                },
            },
            "operations": {
                "clock_pulse": "do_short_pulse",
            },
        },
    },
    "pulses": {
        "ao_cw": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "single": "cw",
            },
        },
        "do_on": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "off",
        },
        "do_short_pulse": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "square",
        },
    },
    "waveforms": {
        "cw": {
            "type": "constant",
            "sample": 0.5,
        },
    },
    "digital_waveforms": {
        "on": {
            "samples": [(1, 0)],
        },
        "off": {
            "samples": [(0, 0)],
        },
        "square": {
            "samples": [(1, 100), (0, 100)],
        },
    },
}

loaded_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "3": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "4": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "5": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "6": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "7": {
                    "offset": 0.0,
                    "delay": 150,
                    "shareable": False,
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
            },
            "analog_inputs": {
                "1": {
                    "offset": 0.0,
                    "gain_db": 0,
                    "shareable": False,
                },
                "2": {
                    "offset": 0.0,
                    "gain_db": 0,
                    "shareable": False,
                },
            },
            "digital_outputs": {
                "1": {
                    "shareable": False,
                    "inverted": False,
                },
                "2": {
                    "shareable": False,
                    "inverted": False,
                },
                "3": {
                    "shareable": False,
                    "inverted": False,
                },
                "4": {
                    "shareable": False,
                    "inverted": False,
                },
                "5": {
                    "shareable": False,
                    "inverted": False,
                },
                "6": {
                    "shareable": False,
                    "inverted": False,
                },
                "7": {
                    "shareable": False,
                    "inverted": False,
                },
                "8": {
                    "shareable": False,
                    "inverted": False,
                },
                "9": {
                    "shareable": False,
                    "inverted": False,
                },
                "10": {
                    "shareable": False,
                    "inverted": False,
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "do1": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 1),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do2": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 2),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do3": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 3),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do4": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 4),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do5": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 5),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do6": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 6),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do7": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 7),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do8": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 8),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do9": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 9),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "do10": {
            "digitalInputs": {
                "chan": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 10),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "on": "do_on",
                "off": "do_off",
            },
        },
        "ao1": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 1),
            },
        },
        "ao2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 2),
            },
        },
        "ao3": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 3),
            },
        },
        "ao4": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 4),
            },
        },
        "ao5": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 5),
            },
        },
        "ao6": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 6),
            },
        },
        "ao7": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 7),
            },
        },
        "ao8": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 8),
            },
        },
        "ao9": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 9),
            },
        },
        "ao10": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 10),
            },
        },
        "laserglow_589_x": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 100000000.0,
            "operations": {
                "cw": "ao_cw",
            },
            "singleInput": {
                "port": ('con1', 1),
            },
        },
        "sig_gen_TEKT_tsg4104a": {
            "digitalInputs": {
                "marker": {
                    "delay": 150,
                    "buffer": 0,
                    "port": ('con1', 7),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "uwave_on": "do_on",
                "uwave_off": "do_off",
            },
        },
        "cobolt_515": {
            "digitalInputs": {
                "marker": {
                    "delay": 150,
                    "buffer": 0,
                    "port": ('con1', 9),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "laser_on": "do_on",
                "laser_off": "do_off",
            },
        },
        "do_sample_clock": {
            "digitalInputs": {
                "marker": {
                    "delay": 150,
                    "buffer": 0,
                    "port": ('con1', 5),
                },
            },
            "digitalOutputs": {},
            "operations": {
                "clock_pulse": "do_short_pulse",
            },
        },
    },
    "pulses": {
        "ao_cw": {
            "length": 1000,
            "waveforms": {
                "single": "cw",
            },
            "operation": "control",
        },
        "do_on": {
            "length": 1000,
            "digital_marker": "on",
            "operation": "control",
        },
        "do_off": {
            "length": 1000,
            "digital_marker": "off",
            "operation": "control",
        },
        "do_short_pulse": {
            "length": 1000,
            "digital_marker": "square",
            "operation": "control",
        },
    },
    "waveforms": {
        "cw": {
            "sample": 0.5,
            "type": "constant",
        },
    },
    "digital_waveforms": {
        "on": {
            "samples": [(1, 0)],
        },
        "off": {
            "samples": [(0, 0)],
        },
        "square": {
            "samples": [(1, 100), (0, 100)],
        },
    },
    "integration_weights": {},
    "mixers": {},
}


