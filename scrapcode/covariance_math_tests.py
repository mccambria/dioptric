# -*- coding: utf-8 -*-
"""
Covariance math tests

Created on January 9th 2025

@author: mccambria
"""


def p_Mi_Mj(mi, mj):
    even_prob = 0.0
    if mi == mj:
        return even_prob / 2
    else:
        return (1 - even_prob) / 2


def p_Yi_I_Mi(yi, mi):
    NV0_ms0_prob = 0.9
    NV0_ms1_prob = 0.7
    if mi == 0:
        if yi == 0:
            return NV0_ms0_prob
        else:
            return 1 - NV0_ms0_prob
    else:
        if yi == 0:
            return NV0_ms1_prob
        else:
            return 1 - NV0_ms1_prob


def mean_Yi():
    term_1 = (p_Mi_Mj(0, 0) + p_Mi_Mj(0, 1)) * p_Yi_I_Mi(1, 0)
    term_2 = (p_Mi_Mj(1, 0) + p_Mi_Mj(1, 1)) * p_Yi_I_Mi(1, 1)
    return term_1 + term_2


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


def integrand(yi, yj, mi, mj):
    return (
        p_Mi_Mj(mi, mj)
        # * f(yi)
        # * f(yj)
        * p_Yi_I_Mi(yi, mi)
        # * p_Yi_I_Mi(yi, negate(mi))
        * p_Yi_I_Mi(yj, mj)
        * (yi - mean_Yi())
        * (yj - mean_Yi())
    )


def integrate():
    total = 0
    for yi in [0, 1]:
        for yj in [0, 1]:
            for mi in [0, 1]:
                for mj in [0, 1]:
                    total += integrand(yi, yj, mi, mj)
    return total


if __name__ == "__main__":
    print(integrate())
