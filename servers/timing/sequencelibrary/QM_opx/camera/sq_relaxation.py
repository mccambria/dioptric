# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence
import utils.common as common
import matplotlib.pyplot as plt


def get_seq(args, num_reps):
    (pol_coords_list, ion_coords_list, tau_ns) = args

    def uwave_macro():
        tau = seq_utils.convert_ns_to_cc(tau_ns)
        buffer = seq_utils.get_widefield_operation_buffer()
        sig_gen_el = seq_utils.get_sig_gen_element()

        qua.wait(tau, sig_gen_el)

        qua.wait(buffer, sig_gen_el)
        qua.align()

    seq = base_sequence.get_seq(pol_coords_list, ion_coords_list, num_reps, uwave_macro)

    seq_ret_vals = []
    return seq, seq_ret_vals


def get_seq(args, num_reps, reference=False):
    (pol_coords_list, ion_coords_list, uwave_ind, tau) = args

    if num_reps == None:
        num_reps = 1

    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    buffer = seq_utils.get_widefield_operation_buffer()
    half_tau = seq_utils.convert_ns_to_cc(tau / 2)

    with qua.program() as seq:
        seq_utils.turn_on_aods()

        def half_rep(no_uwait=False):
            # Polarization
            seq_utils.macro_polarize(pol_coords_list)

            # Microwave sequence
            if not no_uwait:
                qua.wait(tau, sig_gen_el)
            qua.wait(buffer, sig_gen_el)
            qua.align()

            # Ionization
            seq_utils.macro_ionize(ion_coords_list)

            # Readout
            seq_utils.macro_charge_state_readout()

        def one_rep():
            if reference:
                for no_uwait in (False, True):
                    half_rep(no_uwait)
                    qua.align()
                    seq_utils.macro_wait_for_trigger()
            else:
                half_rep(False)
                qua.align()
                seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

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
            1000.0,
            [
                [112.8143831410256, 110.75435400118901],
                [112.79838314102561, 110.77035400118902],
            ],
            "laser_COBO_638",
            200,
            [
                [76.56091979499166, 75.8487161634141],
                [76.30891979499165, 75.96071616341409],
            ],
            "laser_OPTO_589",
            3500.0,
            "sig_gen_STAN_sg394",
            96 / 2,
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
