# -*- coding: utf-8 -*-
"""
Base spin sequence for multi-NV experiments

Created on December 11th, 2023

@author: mccambria
"""


from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
from servers.timing.sequencelibrary.QM_opx import seq_utils
import utils.common as common
import matplotlib.pyplot as plt


def get_seq(
    pol_coords_list,
    ion_coords_list,
    num_reps,
    uwave_macro,
    pol_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
):
    if num_reps == None:
        num_reps = 1

    # Determine how many experiments to run based on len of uwave_macro
    try:
        num_exps_per_rep = len(uwave_macro)
    except Exception as exc:
        num_exps_per_rep = 1

    with qua.program() as seq:
        seq_utils.turn_on_aods()

        def one_exp(exp_ind=None):
            # Polarization
            seq_utils.macro_polarize(pol_coords_list, pol_duration_ns)

            # Custom macro for the microwave sequence here
            if exp_ind is None:
                uwave_macro()
            else:
                exp_uwave_macro = uwave_macro[exp_ind]
                exp_uwave_macro()

            # Ionization
            seq_utils.macro_ionize(ion_coords_list, ion_duration_ns)

            # Readout
            seq_utils.macro_charge_state_readout(readout_duration_ns)

            qua.align()
            seq_utils.macro_wait_for_trigger()

        def one_rep():
            if num_exps_per_rep == 1:
                one_exp()
            else:
                for exp_ind in range(num_exps_per_rep):
                    one_exp(exp_ind)

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

    return seq


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
