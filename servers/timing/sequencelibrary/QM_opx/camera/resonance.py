# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


import numpy
from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import matplotlib.pyplot as plt
from qm import generate_qua_script


def get_seq(args, num_reps):
    (
        pol_laser,
        pol_duration_ns,
        pol_coords_list,
        ion_laser,
        ion_duration_ns,
        ion_coords_list,
        readout_laser,
        readout_duration_ns,
        sig_gen_name,
        uwave_duration_ns,
    ) = args

    if num_reps == None:
        num_reps = 1

    sig_gen_el = f"do_{sig_gen_name}_dm"
    no_uwave = uwave_duration_ns == 0
    if not no_uwave:
        uwave_duration = seq_utils.convert_ns_to_cc(uwave_duration_ns, raise_error=True)
    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:
        seq_utils.turn_on_aods([pol_laser, ion_laser])

        def one_rep():
            # Polarization
            seq_utils.macro_polarize(pol_laser, pol_duration_ns, pol_coords_list)

            # Microwave sequence
            if not no_uwave:
                qua.play("on", sig_gen_el, duration=uwave_duration)
            qua.wait(buffer, sig_gen_el)
            qua.align()

            # Ionization
            seq_utils.macro_ionize(ion_laser, ion_duration_ns, ion_coords_list)

            # Readout
            seq_utils.macro_charge_state_readout(readout_laser, readout_duration_ns)

        # def one_rep():
        #     for half_rep_args in [
        #         [do_polarize_sig, do_ionize_sig],
        #         [do_polarize_ref, do_ionize_ref],
        #     ]:
        #         half_rep(*half_rep_args)
        #         # qua.align()
        #         seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(one_rep, num_reps)
        # seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            "laser_INTE_520",
            10000.0,
            [
                [112.18352668291094, 110.34002169977755],
                [112.07852668291093, 110.84802169977756],
                [110.83652668291093, 111.82802169977755],
                [111.27252668291094, 109.38302169977756],
                [112.20752668291094, 109.49302169977756],
                [112.92852668291094, 110.40002169977755],
                [111.47052668291093, 107.64302169977756],
                [111.55952668291094, 107.65202169977756],
                [112.06852668291094, 107.02702169977756],
                [110.99552668291093, 110.24502169977755],
            ],
            "laser_COBO_638",
            220,
            [
                [75.97049095236301, 75.29800391512516],
                [75.62149095236302, 76.03300391512516],
                [75.03949095236301, 76.68000391512516],
                [75.18649095236302, 74.79700391512516],
                [75.93349095236302, 74.83800391512516],
                [76.54149095236302, 75.46700391512516],
                [75.42949095236301, 73.32500391512517],
                [75.28149095236301, 73.24000391512516],
                [75.80649095236302, 72.78900391512516],
                [75.00549095236302, 75.30800391512516],
            ],
            "laser_OPTO_589",
            35000000.0,
            "sig_gen_STAN_sg394",
            64,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(2000e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
