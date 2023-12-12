# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


import numpy as np
from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence
import utils.common as common
import matplotlib.pyplot as plt


def get_seq(
    args,
    num_reps,
    reference=True,
    pol_duration_ns=None,
    uwave_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
    phase=None,
):
    (pol_coords_list, ion_coords_list, uwave_ind) = args

    if phase is not None:
        i_el, q_el = seq_utils.get_iq_mod_elements(uwave_ind)
        phase_rad = phase * (np.pi / 180)
        i_comp = 0.5 * np.cos(phase_rad)
        q_comp = 0.5 * np.sin(phase_rad)

    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    uwave_duration = seq_utils.convert_ns_to_cc(uwave_duration_ns, raise_error=True)
    buffer = seq_utils.get_widefield_operation_buffer()

    def uwave_macro_sig():
        if uwave_duration is None:
            qua.play("pi_pulse", sig_gen_el)
        else:
            qua.play("on", sig_gen_el, duration=uwave_duration)
        # if phase is not None:
        #     qua.play("off", i_el)
        #     qua.play("on", q_el)
        qua.wait(buffer, sig_gen_el)
        qua.align()

    def uwave_macro_ref():
        pass

    if reference:
        uwave_macro = [uwave_macro_sig, uwave_macro_ref]
    else:
        uwave_macro = uwave_macro_sig

    seq = base_sequence.get_seq(
        pol_coords_list,
        ion_coords_list,
        num_reps,
        uwave_macro,
        pol_duration_ns,
        ion_duration_ns,
        readout_duration_ns,
    )

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
