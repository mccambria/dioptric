# -*- coding: utf-8 -*-
"""
Base spin sequence for widefield experiments with many spatially resolved NV centers.
Accompanies base routine

Created on December 11th, 2023

@author: mccambria
"""

from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    pol_coords_list,
    ion_coords_list,
    spin_flip_ind_list,
    uwave_ind,
    uwave_macro,
    step_vals=None,
    num_reps=1,
    pol_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
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

    if num_reps is None:
        num_reps = 1

    # Construct the list of experiments to run
    if not isinstance(uwave_macro, list):
        uwave_macro = [uwave_macro]
    if reference:

        def ref_exp(step_val):
            pass

        uwave_macro.append(ref_exp)
    num_exps_per_rep = len(uwave_macro)

    ### QUA stuff

    seq_utils.init()
    step_val = qua.declare(int)

    def one_exp(exp_ind):
        seq_utils.macro_polarize(pol_coords_list, pol_duration_ns)
        uwave_macro[exp_ind](step_val)
        # Always look at ms=0 counts for the reference
        exp_spin_flip_ind_list = spin_flip_ind_list if exp_ind == 0 else None
        seq_utils.macro_scc(
            ion_coords_list, exp_spin_flip_ind_list, uwave_ind, ion_duration_ns
        )
        seq_utils.macro_charge_state_readout(readout_duration_ns)
        seq_utils.macro_wait_for_trigger()

    def one_rep():
        for exp_ind in range(num_exps_per_rep):
            one_exp(exp_ind)

    def one_step():
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        seq_utils.macro_pause()

    if step_vals is None:
        one_step()
    else:
        with qua.for_each_(step_val, step_vals):
            one_step()
