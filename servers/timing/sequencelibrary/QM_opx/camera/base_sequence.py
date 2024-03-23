# -*- coding: utf-8 -*-
"""
Base spin sequence for widefield experiments with many spatially resolved NV centers.
Accompanies base routine

Created on December 11th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils.constants import IonPulseType


def get_seq(
    pol_coords_list,
    ion_coords_list,
    num_reps,
    uwave_macro,
    pol_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
    setup_macro=None,
    reference=True,
    ion_pulse_type=IonPulseType.SCC,
):
    """Base spin sequence for widefield experiments with many spatially resolved NV
    centers. Accompanies base routine

    Parameters
    ----------
    pol_coords_list : list(float)
        List of polarization coordinates to target -
        returned by widefield.get_base_scc_seq_args(nv_list)
    ion_coords_list : list(float)
        List of ionization coordinates to target -
        returned by widefield.get_base_scc_seq_args(nv_list)
    num_reps : int
        Number of repetitions to do
    uwave_macro : function or list(function)
        QUA macro describing the microwave pulse sequence. If a list, then
        run multiple experiments per rep - the first macro will be run for
        the first experiment, the second macro for the second experiment, etc.
    pol_duration_ns : int, optional
        Duration of polarization pulse in ns, by default pulls from config
    ion_duration_ns : int, optional
        Duration of ionization pulse in ns, by default pulls from config
    readout_duration_ns : int, optional
        Duration of readout pulse in ns, by default pulls from config
    setup_macro : function, optional
        QUA macro describing any setup sequence we may want to run. Called
        at the very beginning even before turning on AODs. By default None
    reference : bool, optional
        Whether to include a reference experiment in which no microwaves
        are applied, by default True

    Returns
    -------
    _type_
        _description_
    """
    if num_reps is None:
        num_reps = 1

    # Construct the list of experiments to run
    if not isinstance(uwave_macro, list):
        uwave_macro = [uwave_macro]
    if reference:

        def ref_exp():
            pass

        uwave_macro.append(ref_exp)
    num_exps_per_rep = len(uwave_macro)
    # uwave_macro = uwave_macro[::-1]  # MCC

    with qua.program() as seq:
        if setup_macro is not None:
            uwave_macro_args = setup_macro()
        else:
            uwave_macro_args = []

        seq_utils.turn_on_aods()

        def one_exp(exp_ind=None):
            # Polarization
            seq_utils.macro_polarize(pol_coords_list, pol_duration_ns)

            # Custom macro for the microwave sequence here
            qua.align()
            if exp_ind is None:
                uwave_macro(*uwave_macro_args)
            else:
                exp_uwave_macro = uwave_macro[exp_ind]
                exp_uwave_macro(*uwave_macro_args)

            # Ionization
            seq_utils.macro_ionize(ion_coords_list, ion_duration_ns, ion_pulse_type)

            # Readout
            seq_utils.macro_charge_state_readout(readout_duration_ns)

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

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
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
