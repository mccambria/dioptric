# -*- coding: utf-8 -*-
"""
QM OPX sequence utils

Created June 25th, 2023

@author: mccambria
"""


from qm.qua import declare, assign, infinite_loop_, while_


def handle_reps(one_rep, num_reps):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop

    Parameters
    ----------
    one_rep : function
        QUA "macro" to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    """

    if num_reps == -1:
        with infinite_loop_():
            one_rep()
    else:
        ind = declare(int, value=0)
        with while_(ind < num_reps):
            one_rep()
            assign(ind, ind + 1)
