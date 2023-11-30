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
    uwave_duration = seq_utils.convert_ns_to_cc(uwave_duration_ns)
    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:
        seq_utils.turn_on_aods([pol_laser, ion_laser])

        def one_rep():
            # Polarization
            seq_utils.macro_polarize(pol_laser, pol_duration_ns, pol_coords_list)

            # Microwave sequence
            if uwave_duration > 0:
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
            [[112.164, 109.832]],
            "laser_COBO_638",
            1000.0,
            [[75.97, 75.202]],
            "laser_OPTO_589",
            50000.0,
            False,
            True,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(500e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
