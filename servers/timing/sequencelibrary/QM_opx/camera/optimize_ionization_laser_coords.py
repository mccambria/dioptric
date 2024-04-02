# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils import common
from utils import tool_belt as tb
from utils.constants import IonPulseType, LaserKey


def get_seq(pol_coords_list, ion_coords_list, num_reps):
    if num_reps is None:
        num_reps = 1

    ion_pulse_type = IonPulseType.ION
    green_laser = tb.get_laser_name(LaserKey.POLARIZATION)
    red_laser = tb.get_laser_name(LaserKey.IONIZATION)

    with qua.program() as seq:
        seq_utils.init_cache()
        seq_utils.turn_on_aods(
            [green_laser, red_laser], aod_suffices=["charge_pol", "opti"]
        )

        def one_rep():
            # Charge polarization with green, spin polarization with yellow
            seq_utils.macro_polarize(pol_coords_list)

            # Ionization
            seq_utils.macro_ionize(ion_coords_list, ion_pulse_type=ion_pulse_type)

            # Readout
            seq_utils.macro_charge_state_readout()

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
            5000.0,
            "laser_OPTO_589",
            True,
            "laser_INTE_520",
            [111.202, 109.801],
            1000,
            False,
            "laser_COBO_638",
            [75, 75],
            2000.0,
        ]
        seq, seq_ret_vals = get_seq(*args, 5)

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
