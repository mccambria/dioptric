# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import LaserKey


def get_seq(
    pol_coords_list,
    ion_coords_list,
    uwave_ind,
    aod_voltages,
    num_reps=1,
):
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    buffer = seq_utils.get_widefield_operation_buffer()
    laser_name = tb.get_laser_name(LaserKey.POLARIZATION)

    if num_reps is None:
        num_reps = 1

    def uwave_macro_sig():
        qua.play("pi_pulse", sig_gen_el)
        qua.wait(buffer, sig_gen_el)

    def uwave_macro_ref():
        pass

    opx_config = common.get_opx_config_dict()
    base_voltage = opx_config["waveforms"]["green_aod_cw"]
    num_steps = len(aod_voltages)
    aod_amps = [aod_voltages[ind] / base_voltage for ind in range(num_steps)]

    uwave_macro = [uwave_macro_sig, uwave_macro_ref]
    num_exps_per_rep = len(uwave_macro)

    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:
        aod_amp = qua.declare(qua.fixed)

        def one_exp(exp_ind):
            seq_utils.turn_on_aods()
            # Charge polarization with green
            seq_utils.macro_polarize(pol_coords_list)

            # Custom macro for the microwave sequence here
            qua.align()
            exp_uwave_macro = uwave_macro[exp_ind]
            exp_uwave_macro()

            seq_utils.turn_on_aods(laser_names=[laser_name], amps=[aod_amp])

            # Repolarization pulse
            seq_utils.macro_pulse(laser_name, pol_coords_list[0], pulse_name="polarize")

            # Ionization
            seq_utils.macro_ionize(ion_coords_list)

            # Readout
            seq_utils.macro_charge_state_readout()

            seq_utils.macro_wait_for_trigger()

        def one_rep():
            for exp_ind in range(num_exps_per_rep):
                one_exp(exp_ind)

        def one_step():
            seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

            # Make sure everything is off before pausing for the next step
            qua.align()
            qua.wait(buffer)
            qua.pause()

        with qua.for_each_(aod_amp, aod_amps):
            one_step()

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [109.05560372660722, 110.77022466032236],
            ],
            [
                [73.78442169604547, 75.67270342527479],
            ],
            0,
            "laser_COBO_638",
            [[73.5, 75.5], [73.5, 75.6], [73.5, 75.7], [73.5, 75.8]],
            1,
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
