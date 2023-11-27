# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

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
        diff_polarize,
        diff_ionize,
    ) = args

    if num_reps == None:
        num_reps = 1

    if diff_polarize and not diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = False
        do_ionize_sig = False
        do_ionize_ref = False
    elif not diff_polarize and diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = True
        do_ionize_sig = True
        do_ionize_ref = False

    with qua.program() as seq:
        seq_utils.turn_on_aods([pol_laser, ion_laser])

        def half_rep(do_polarize_sub, do_ionize_sub):
            # Polarization
            seq_utils.macro_polarize(
                pol_laser, pol_duration_ns, pol_coords_list, not do_polarize_sub
            )

            # Ionization
            seq_utils.macro_ionize(
                ion_laser, ion_duration_ns, ion_coords_list, not do_ionize_sub
            )

            # Readout
            seq_utils.macro_charge_state_readout(readout_laser, readout_duration_ns)

        def one_rep():
            for half_rep_args in [
                [do_polarize_sig, do_ionize_sig],
                [do_polarize_ref, do_ionize_ref],
            ]:
                half_rep(*half_rep_args)

                # qua.align()
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
            10000.0,
            [
                [110, 109.51847988358679],
                [0.001, 110.70156405156148],
            ],
            "laser_COBO_638",
            1000.0,
            [
                [75.42725784791932, 75.65982013416432],
                [75.98725784791932, 74.74382013416432],
            ],
            "laser_OPTO_589",
            50000000.0,
            False,
            True,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
