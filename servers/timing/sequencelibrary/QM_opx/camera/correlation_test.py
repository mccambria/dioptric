# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence


def get_seq(args, num_reps):
    (pol_coords_list, ion_coords_list, tau_ns) = args

    tau = seq_utils.convert_ns_to_cc(tau_ns, allow_zero=True)
    short_wait = seq_utils.convert_ns_to_cc(100)
    buffer = seq_utils.get_widefield_operation_buffer()
    sig_gen_el = seq_utils.get_sig_gen_element()
    i_el, q_el = seq_utils.get_iq_mod_elements()

    random = qua.Random()
    rand_phase = qua.declare(qua.fixed)

    def uwave_macro():
        qua.play("pi_pulse", i_el)
        qua.play("pi_on_2_pulse", sig_gen_el)

        qua.wait(short_wait)

        if tau != 0:
            qua.assign(rand_phase, random.rand_fixed())
            qua.play("pi_pulse" * qua.amp(qua.Math.cos2pi(rand_phase)), i_el)
            qua.play("pi_pulse" * qua.amp(qua.Math.sin2pi(rand_phase)), q_el)
            qua.play("pi_pulse", sig_gen_el, duration=tau)

        qua.wait(short_wait)

        qua.play("pi_pulse", i_el)
        qua.play("pi_on_2_pulse", sig_gen_el)

        qua.wait(buffer, sig_gen_el)

    seq = base_sequence.get_seq(pol_coords_list, ion_coords_list, num_reps, uwave_macro)

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
            [
                [111.4994186339929, 108.79019926783882],
                [110.7254186339929, 109.27119926783882],
                [111.21841863399291, 108.10619926783882],
                [111.61041863399291, 107.73419926783882],
                [112.26641863399291, 108.27919926783882],
                [112.62341863399291, 108.44219926783882],
                [112.5374186339929, 108.63419926783882],
                [112.1534186339929, 109.32819926783883],
                [111.83141863399291, 111.34519926783882],
                [109.84041863399291, 110.76519926783882],
            ],
            [
                [75.60717320911249, 74.33558456443815],
                [74.88417320911249, 74.52458456443814],
                [75.1231732091125, 73.47458456443815],
                [75.90117320911249, 73.64858456443815],
                [76.02317320911249, 73.57858456443815],
                [76.22917320911249, 73.79558456443814],
                [76.40517320911249, 74.01058456443815],
                [75.96417320911249, 74.60258456443815],
                [75.5491732091125, 76.43458456443815],
                [74.14917320911249, 75.88658456443815],
            ],
            1000.0,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(1000e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
