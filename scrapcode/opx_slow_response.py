# -*- coding: utf-8 -*-
"""
Minimal working example to show slow user time of OPX for simple sequences

Created on November 5th, 2023

@author: mccambria
"""


from qm import QuantumMachinesManager
from qm.qua import program, play
from qm import generate_qua_script
import time

default_len = 80

opx_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": 0},
                2: {"offset": 0.0, "delay": 0},
                3: {"offset": 0.0, "delay": 0},
                4: {"offset": 0.0, "delay": 0},
                5: {"offset": 0.0, "delay": 0},
                6: {"offset": 0.0, "delay": 0},
                7: {"offset": 0.0, "delay": 0},
                8: {"offset": 0.0, "delay": 0},
                9: {"offset": 0.0, "delay": 0},
                10: {"offset": 0.0, "delay": 0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
                9: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        },
    },
    "elements": {
        "ao_laser_OPTO_589_am": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "operations": {"on": "ao_cw", "off": "ao_off"},
        },
        "do_camera_trigger": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "sticky": {"analog": True, "digital": True, "duration": 160},
            "operations": {"on": "do_on", "off": "do_off"},
        },
    },
    "pulses": {
        "ao_cw": {
            "operation": "control",
            "length": default_len,
            "waveforms": {"single": "cw"},
        },
        "ao_off": {
            "operation": "control",
            "length": default_len,
            "waveforms": {"single": "off"},
        },
        "do_on": {
            "operation": "control",
            "length": default_len,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": default_len,
            "digital_marker": "off",
        },
    },
    "waveforms": {
        "cw": {"type": "constant", "sample": 0.5},
        "off": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "on": {"samples": [(1, 0)]},
        "off": {"samples": [(0, 0)]},
    },
}


def qua_program():
    """Readout with a camera for 1 ms"""
    laser_element = "ao_laser_OPTO_589_am"
    camera_element = "do_camera_trigger"
    readout_cc = int(1e6 / 4)
    with program() as seq:
        play("on", laser_element, duration=readout_cc)
        play("on", camera_element, duration=readout_cc)
        play("off", camera_element)
    return seq


def minimal_working_example():
    ip_address = "192.168.0.117"
    qmm = QuantumMachinesManager(ip_address)

    try:
        opx = qmm.open_qm(opx_config)

        seq = qua_program()

        sourceFile = open("opx_slow_response-serialized.py", "w")
        print(generate_qua_script(seq, opx_config), file=sourceFile)
        sourceFile.close()

        start = time.time()
        program_id = opx.compile(seq)
        end = time.time()
        print(f"compile time: {round(end - start, 3)}")
        print()

        for ind in range(20):
            start = time.time()
            pending_job = opx.queue.add_compiled(program_id)
            end = time.time()
            print(f"add_compiled time: {round(end - start, 3)}")
            start = end
            job = pending_job.wait_for_execution()
            end = time.time()
            print(f"wait_for_execution time: {round(end - start, 3)}")
            start = end
            while job.status != "completed":
                pass
            end = time.time()
            print(f"job completed time: {round(end - start, 3)}")
            print()

    finally:
        qmm.close_all_quantum_machines()
        qmm.close()


if __name__ == "__main__":
    minimal_working_example()
