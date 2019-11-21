# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:47:03 2019

@author: matth
"""


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy
from analysis import rotation_dq_sq_ratio_v2 as mat_el_calc
from analysis import extract_hamiltonian
from scipy import integrate
from numpy import pi


def empirical(splitting, coeff, offset):
    return (coeff / splitting**2) + offset

def empirical2(splitting, coeff, offset):
    return ((coeff / splitting) + offset)**2

def empirical3_base(hamiltonian_params):
    def empirical3_local(splitting, coeff, offset):
        noise_func = mat_el_calc.calc_Pi_factor_surface
        mat_els = []
        for val in splitting:
            mag_B = extract_hamiltonian.find_mag_B_splitting(val,
                                                 *hamiltonian_params)
            mat_el, err = integrate.dblquad(noise_func, 0, 2*pi,
                                        lambda x: 0, lambda x: pi,
                                        args=(mag_B, hamiltonian_params, 2))
            mat_els.append(mat_el)
        mat_els = numpy.array(mat_els)
        return ((coeff / splitting**2) + offset) * mat_els
        # return mat_els
    return empirical3_local


def fit_data(name, res_descs, meas_splittings, meas_gammas, error_gammas):

    empirical_popt, empirical_pcov = curve_fit(empirical,
                                        meas_splittings, meas_gammas,
                                        sigma=error_gammas, p0=(0.03, 0.5))
    print(empirical_popt)
    empirical2_popt, empirical2_pcov = curve_fit(empirical2,
                                        meas_splittings, meas_gammas,
                                        sigma=error_gammas, p0=(0.15, 0.6))
    print(empirical2_popt)
    hamiltonian_params = extract_hamiltonian.main(name, res_descs)
    empirical3 = empirical3_base(hamiltonian_params)
    empirical3_popt, empirical3_pcov = curve_fit(empirical3,
                                        meas_splittings, meas_gammas,
                                        sigma=error_gammas, p0=(1.4e-5, 1.7e-4))
    print(empirical3_popt)

    splitting_logspace = numpy.logspace(-1.8, 0.3, 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.set_tight_layout(True)
    ax.loglog(splitting_logspace,
              empirical(splitting_logspace, *empirical_popt))
    ax.plot(splitting_logspace,
            empirical2(splitting_logspace, *empirical2_popt))
    ax.plot(splitting_logspace,
            empirical3(splitting_logspace, *empirical3_popt))
    ax.scatter(meas_splittings, meas_gammas)
    ax.set_xlabel('Splitting (GHz)', fontsize=18)
    ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)


if __name__ == '__main__':

    # Here all the data points used in the paper along with the actual
    # measured resonances accompanying the points

    ##############################

    # name = 'nv1_2019_05_10'  # NV1
    # res_descs = [[0.0, 2.8544, 2.8739],
    #               [None, 2.8554, 2.8752],
    #               [None, 2.8512, 2.8790],
    #               # [None, 2.8520, 2.8800],  # not used in paper
    #               [None, 2.8503, 2.8792],
    #               # [None, 2.8536, 2.8841],  # not used in paper
    #               [None, 2.8396, 2.8917],
    #               [None, 2.8496, 2.8823],
    #               # [None, 2.8198, 2.9106],  # misaligned ref
    #               [None, 2.8166, 2.9144],
    #               [None, 2.8080, 2.9240],
    #               [None, 2.7357, 3.0037],
    #               # [None, 2.7374, 3.0874],  # misaligned
    #               # [None, 2.6310, 3.1547],  # misaligned ref for prev
    #               [None, 2.6061, 3.1678],
    #               # [None, 2.6055, 3.1691],  # repeat of previous
    #               [None, 2.4371, 3.4539],  # 0,-1 and 0,+1 omegas
    #               # [None, 2.4381, 3.4531],   # retake 0,-1 and 0,+1 omegas
    #               ]
    # misaligned_res_desc = [None, 2.7374, 3.0874]
    # misaligned_ref_res_desc = [None, 2.6310, 3.1547]
    # meas_splittings = numpy.array([19.5, 19.8, 27.7, 28.9, 32.7, 51.8,
    #                     97.8, 116, 268, 350, 561.7, 1016.8])
    # meas_gammas = numpy.array([58.3, 117, 64.5, 56.4, 42.6, 13.1, 3.91,
    #                                 4.67, 1.98, 1.57, 0.70, 0.41])
    # error_gammas = numpy.array([1.4, 4, 1.4, 1.3, 0.9, 0.2, 0.1,
    #                             0.11, 0.1, 0.12, 0.05, 0.05])

    ##############################

    name = 'nv2_2019_04_30'  # NV2
    res_descs = [
                  [0.0, 2.8582, 2.8735],
                  # [None, 2.8512, 2.8804],
                  # [None, 2.8435, 2.8990],
                  # [None, 2.8265, 2.9117],
                  # [None, 2.7726, 3.0530],
                  # [None, 2.7738, 3.4712],
                   [None, 2.8507, 2.8798],  # Take 2 starts here
                   [None, 2.8434, 2.8882],
                   [None, 2.8380, 2.8942],
                   [None, 2.8379, 2.8948],
                   # [None, 2.8308, 2.9006],  # not used in paper
                   # [None, 2.8228, 2.9079],  # not used in paper
                   [None, 2.8155, 2.9171],
                  ]
    meas_splittings = numpy.array([15.3, 29.2, 45.5, 85.2, 280.4, 697.4,
                                    29.1, 44.8, 56.2, 56.9, 101.6])
    meas_gammas = numpy.array([124, 31.1, 8.47, 2.62, 0.443, 0.81, 20.9,
                                6.43, 3.64, 3.77,  1.33])
    error_gammas = numpy.array([3, 0.4, 0.11, 0.05, 0.014, 0.06,
                                0.3, 0.12, 0.08, 0.09, 0.05])

    ##############################

    # name = 'NV16_2019_07_25'  # NV3
    # res_descs = [[0.0, 2.8593, 2.8621],  # no T1
    #               [None, 2.8519, 2.8690],
    #               [None, 2.8460, 2.8746],
    #               [None, 2.8337, 2.8867],
    #               [None, 2.8202, 2.9014],
    #               [None, 2.8012, 2.9292],
    #               [None, 2.7393, 3.0224],
    #               [None, 2.6995, 3.1953],
    #               [None, 2.5830, 3.3290],
    #               ]
    # meas_splittings = numpy.array([17.1, 28.6, 53.0, 81.2,
    #                                 128.0, 283.1, 495.8, 746])
    # meas_gammas = numpy.array([108, 90, 26.2, 17.5, 11.3, 5.6, 3.7, 2.8])
    # error_gammas = numpy.array([10, 5, 0.9, 0.6, 0.4, 0.3, 0.4, 0.3])

    ##############################

    # name = 'NV0_2019_06_06'  # NV4
    # res_descs = [
    #               # [0.0, 2.8547, 2.8793],  # old zero field
    #               [0.0, 2.8556, 2.8790],
    #               [None, 2.8532, 2.8795],
    #               [None, 2.8494, 2.8839],
    #               [None, 2.8430, 2.8911],
    #               [None, 2.8361, 2.8998],
    #               [None, 2.8209, 2.9132],
    #               [None, 2.7915, 2.9423],
    #               [None, 2.7006, 3.0302],
    #               [None, 2.4244, 3.3093],
    #               # [None, 2.4993, 3.5798],  # misaligned
    #               [None, 2.2990, 3.4474],
    #               ]
    # meas_splittings = numpy.array([23.4, 26.2, 36.2, 48.1, 60.5, 92.3, 150.8,
    #                                 329.6, 884.9, 1080.5, 1148.4])
    # meas_gammas = numpy.array([34.5, 29.0, 20.4, 15.8, 9.1, 6.4, 4.08,
    #                             1.23, 0.45, 0.69, 0.35])
    # error_gammas = numpy.array([1.3, 1.1, 0.5, 0.3, 0.3, 0.1,
    #                             0.15, 0.07, 0.03, 0.12, 0.03])

    ##############################

    # name = 'nv13_2019_06_10'  # NV5
    # res_descs = [[0.0, 2.8365, 2.8446],  # no T1
    #               [None, 2.8363, 2.8472],
    #               [None, 2.8289, 2.8520],
    #               # [None, 2.8266, 2.8546],  # not used in paper
    #               # [None, 2.8262, 2.8556],  # not used in paper
    #               [None, 2.8247, 2.8545],
    #               [None, 2.8174, 2.8693],
    #               [None, 2.8082, 2.8806],
    #               [None, 2.7948, 2.9077],
    #               [None, 2.7857, 2.9498],
    #               [None, 2.7822, 3.0384],
    #               ]
    # meas_splittings = numpy.array([10.9, 23.1, 29.8, 51.9,
    #                                 72.4, 112.9, 164.1, 256.2])
    # meas_gammas = numpy.array([240, 62, 19.3, 17.7, 16.2, 12.1, 5.6, 2.1])
    # error_gammas = numpy.array([25, 8,  1.1, 1.4, 1.1, 0.9, 0.5, 0.3])

    ##############################

    meas_splittings /= 1000
    # fit_data(name, res_descs, meas_splittings, meas_gammas, error_gammas)
    # print(len(res_descs))
    # print(len(meas_splittings))
    # print(len(meas_gammas))
    # print(len(error_gammas))
    # mat_el_calc.main_plot_paper(name, res_descs, meas_splittings, meas_gammas)
    popt = extract_hamiltonian.main(name, res_descs)
    print((180/pi) * popt[0])
    # angles = extract_hamiltonian.extract_rotated_hamiltonian(name, res_descs,
    #                             misaligned_ref_res_desc, misaligned_res_desc)
    # print((180/pi) * angles[0])
