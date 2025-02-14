# -*- coding: utf-8 -*-
"""
Covariance math tests

Created on January 9th 2025

@author: mccambria
"""

import sys

import numpy as np

even_spin_prob = 1
NV0_ms0_prob = 0.9
NV0_ms1_prob = 0.7
NVn_ms0_prob = 1 - NV0_ms0_prob
NVn_ms1_prob = 1 - NV0_ms1_prob


def p_Mi_Mj(mi, mj):
    if mi == mj:
        return even_spin_prob / 2
    else:
        return (1 - even_spin_prob) / 2


def p_Yi_I_Mi(yi, mi):
    if mi == 0:
        if yi == 0:
            return NV0_ms0_prob
        else:
            return NVn_ms0_prob
    else:
        if yi == 0:
            return NV0_ms1_prob
        else:
            return NVn_ms1_prob


def mean_Yi():
    return 0.5 * (NVn_ms0_prob + NVn_ms1_prob)


def f(yi):
    if yi == 0:
        return -1
    else:
        return +1


def negate(mi):
    if mi == 0:
        return 1
    else:
        return 0


def A_plus(yi, yj, mi, mj):
    return (
        p_Mi_Mj(mi, mj)
        * p_Yi_I_Mi(yi, mi)
        * p_Yi_I_Mi(yj, mj)
        * (yi - mean_Yi())
        * (yj - mean_Yi())
    )


def A_minus(yi, yj, mi, mj):
    return (
        p_Mi_Mj(mi, mj)
        * p_Yi_I_Mi(yi, mi)
        * p_Yi_I_Mi(yj, negate(mj))
        * (yi - mean_Yi())
        * (yj - mean_Yi())
    )


def cov_p_plus(yi, yj, mi, mj):
    return (
        p_Mi_Mj(mi, mj)
        * f(yi)
        * f(yj)
        * p_Yi_I_Mi(yi, mi)
        * p_Yi_I_Mi(yj, mj)
        # * (yi - mean_Yi())
        # * (yj - mean_Yi())
        * (yi)
        * (yj)
    )


def cov_p_minus(yi, yj, mi, mj):
    return (
        p_Mi_Mj(mi, mj)
        * f(yi)
        * f(yj)
        * p_Yi_I_Mi(yi, mi)
        * p_Yi_I_Mi(yj, negate(mj))
        * (yi - mean_Yi())
        * (yj - mean_Yi())
    )


def ref_00(yi, yj, mi, mj):
    if mi == 0 and mj == 0:
        return (
            f(yi)
            * f(yj)
            * p_Yi_I_Mi(yi, mi)
            * p_Yi_I_Mi(yj, mj)
            # * (yi - NVn_ms0_prob)
            # * (yj - NVn_ms0_prob)
            * (yi)
            * (yj)
        )
    else:
        return 0


def ref_11(yi, yj, mi, mj):
    if mi == 1 and mj == 1:
        return (
            f(yi)
            * f(yj)
            * p_Yi_I_Mi(yi, mi)
            * p_Yi_I_Mi(yj, mj)
            # * (yi - NVn_ms1_prob)
            # * (yj - NVn_ms1_prob)
            * (yi)
            * (yj)
        )
    else:
        return 0


def integrate(integrand):
    total = 0
    for yi in [0, 1]:
        for yj in [0, 1]:
            for mi in [0, 1]:
                for mj in [0, 1]:
                    total += integrand(yi, yj, mi, mj)
    return total


if __name__ == "__main__":
    # print(p_Mi_Mj(1, 0))
    # sys.exit()
    var_r = 0.1
    # +/+ and +/-
    print(integrate(A_plus) - integrate(A_minus))
    print(
        (integrate(A_plus) + var_r * integrate(cov_p_plus))
        - (integrate(A_minus) + var_r * integrate(cov_p_minus))
    )
    # +/+ and 0/0
    print()
    print(integrate(A_plus))
    print(integrate(ref_00) / (NVn_ms0_prob * NV0_ms0_prob))
    print(integrate(ref_11) / (NVn_ms1_prob * NV0_ms1_prob))
    print()
    print(
        (integrate(A_plus) + var_r * integrate(cov_p_plus)) - var_r * integrate(ref_00)
    )
    print(
        (integrate(A_plus) + var_r * integrate(cov_p_plus)) - var_r * integrate(ref_11)
    )
    print(
        (integrate(A_plus) + var_r * integrate(cov_p_plus))
        - var_r * 0.5 * (integrate(ref_00) + integrate(ref_11))
    )
    print(integrate(cov_p_plus) - 0.5 * (integrate(ref_00) + integrate(ref_11)))
