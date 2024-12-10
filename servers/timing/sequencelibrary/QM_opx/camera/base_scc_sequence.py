# -*- coding: utf-8 -*-
"""
Base spin sequence for widefield experiments with many spatially resolved NV centers.
Accompanies base routine

Created on December 11th, 2023

@author: mccambria
"""

import numpy as np
from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    base_scc_seq_args,
    uwave_macro,
    step_vals=None,
    num_reps=1,
    scc_duration_override=None,
    scc_amp_override=None,
    reference=True,
):
    """Base spin sequence as a QUA macro for widefield experiments with many
    spatially resolved NV centers. Accompanies base routine

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

    ### Non-QUA stuff

    (
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_do_target_list,
        uwave_ind_list,
    ) = base_scc_seq_args

    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]

    if num_reps is None:
        num_reps = 1

    # Construct the list of experiments to run
    if not isinstance(uwave_macro, list):
        uwave_macro = [uwave_macro]
    if reference:
        uwave_macro.append(lambda uwave_ind_list, step_val: None)
    # if reference:

    #     def ref_exp(uwave_ind_list, step_val):
    #         pass

    #     uwave_macro.append(ref_exp)

    num_exps_per_rep = len(uwave_macro)
    num_nvs = len(pol_coords_list)

    def macro_scc_sub(do_target_list=None):
        seq_utils.macro_scc(
            scc_coords_list,
            scc_duration_list,
            scc_amp_list,
            scc_duration_override,
            scc_amp_override,
            do_target_list,
        )

    ### QUA stuff

    seq_utils.init(num_nvs)

    if step_vals is not None and len(step_vals) > 0:
        step_vals = np.array(step_vals)
        step_int_check = [el.is_integer() for el in step_vals]
        if all(step_int_check):
            step_val = qua.declare(int)
        else:
            step_val = qua.declare(qua.fixed)
    else:
        step_val = qua.declare(int)

    def one_exp(rep_ind, exp_ind):
        # exp_ind = num_exps_per_rep - 1  # MCC
        seq_utils.macro_polarize(
            pol_coords_list, duration_list=pol_duration_list, amp_list=pol_amp_list
        )
        qua.align()
        uwave_macro[exp_ind](uwave_ind_list, step_val)

        # Check if this is the automatically included reference experiment
        ref_exp = reference and exp_ind == num_exps_per_rep - 1
        # Signal experiment
        if not ref_exp:
            if spin_flip_do_target_list is None or True not in spin_flip_do_target_list:
                macro_scc_sub()
            else:
                spin_flip_do_not_target_list = [
                    not (val) for val in spin_flip_do_target_list
                ]
                macro_scc_sub(spin_flip_do_not_target_list)
                seq_utils.macro_pi_pulse(uwave_ind_list)
                macro_scc_sub(spin_flip_do_target_list)
        # Reference experiment
        else:
            # "Dual-rail" referencing: measure ms=0 for even reps, and ms=+/-1
            # for odd by applying an extra pi pulse just before SCC
            with qua.if_(qua.Cast.unsafe_cast_bool(rep_ind)):
                seq_utils.macro_pi_pulse(uwave_ind_list)
            macro_scc_sub()

        seq_utils.macro_charge_state_readout()
        seq_utils.macro_wait_for_trigger()

    def one_rep(rep_ind=0):
        for exp_ind in range(num_exps_per_rep):
            one_exp(rep_ind, exp_ind)

    def one_step():
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        seq_utils.macro_pause()

    # Handle duration and/or amplitude overrides
    if step_vals is None:
        one_step()
    else:
        if scc_duration_override is not None and scc_amp_override is not None:
            duration_steps = [seq_utils.convert_ns_to_cc(el) for el in step_vals[:, 0]]
            amp_steps = step_vals[:, 1]
            with qua.for_each_(scc_duration_override, duration_steps):
                with qua.for_each_(scc_amp_override, amp_steps):
                    one_step()
        elif scc_duration_override is not None:
            with qua.for_each_(scc_duration_override, step_vals):
                one_step()
        elif scc_amp_override is not None:
            with qua.for_each_(scc_amp_override, step_vals):
                one_step()
        else:
            with qua.for_each_(step_val, step_vals):
                one_step()
