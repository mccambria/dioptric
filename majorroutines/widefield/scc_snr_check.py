# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on December 6th, 2023

@author: mccambria
"""

import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def process_and_plot(data):
    threshold = True
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    if threshold:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts, dynamic_thresh=True
        )

    ### Report the results

    # Include this block if the ref shots measure both ms=0 and ms=+/-1
    # avg_sig_counts, avg_sig_counts_ste, norms = widefield.average_counts(
    #     sig_counts, ref_counts
    # )
    # norms_ms0_newaxis = norms[0][:, np.newaxis]
    # norms_ms1_newaxis = norms[1][:, np.newaxis]
    # contrast = norms_ms1_newaxis - norms_ms0_newaxis
    # norm_counts = (avg_sig_counts - norms_ms0_newaxis) / contrast
    # norm_counts_ste = avg_sig_counts_ste / contrast

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    # There's only one point, so only consider that
    step_ind = 0
    avg_sig_counts = avg_sig_counts[:, step_ind]
    avg_sig_counts_ste = avg_sig_counts_ste[:, step_ind]
    avg_ref_counts = avg_ref_counts[:, step_ind]
    avg_ref_counts_ste = avg_ref_counts_ste[:, step_ind]
    avg_snr = avg_snr[:, step_ind]
    avg_snr_ste = avg_snr_ste[:, step_ind]
    avg_contrast = avg_contrast[:, step_ind]
    avg_contrast_ste = avg_contrast_ste[:, step_ind]

    # Print
    print(avg_snr.tolist())
    print(f"Median SNR: {np.median(avg_snr)}")
    return

    ### Plot

    # Normalized counts bar plots
    # fig, ax = plt.subplots()
    # for ind in range(num_nvs):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(ax, nv_num, norm_counts[ind], yerr=norm_counts_ste[ind])
    # ax.set_xlabel("NV index")
    # ax.set_ylabel("Contrast")

    # SNR bar plots
    # figsize = kpl.figsize
    # figsize[1] *= 1.5
    # counts_fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # snr_fig, ax = plt.subplots()
    # for ind in range(len(nv_list)):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(
    #         axes_pack[0], nv_num, avg_ref_counts[ind], yerr=avg_ref_counts_ste[ind]
    #     )
    #     kpl.plot_bars(
    #         axes_pack[1], nv_num, avg_sig_counts[ind], yerr=avg_sig_counts_ste[ind]
    #     )
    #     kpl.plot_bars(ax, nv_num, avg_snr[ind], yerr=avg_snr_ste[ind])
    # axes_pack[0].set_xlabel("NV index")
    # ax.set_xlabel("NV index")
    # axes_pack[0].set_ylabel("NV- | prep in ms=0")
    # axes_pack[1].set_ylabel("NV- | prep in ms=1")
    # ax.set_ylabel("SNR")
    # return counts_fig, snr_fig

    # SNR histogram
    fig, ax = plt.subplots()
    kpl.histogram(ax, avg_snr, kpl.HistType.STEP, nbins=10)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of occurrences")

    # SNR vs red frequency
    coords_key = "laser_COBO_638_aod"
    distances = []
    for nv in nv_list:
        coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        dist = np.sqrt((90 - coords[0]) ** 2 + (90 - coords[1]) ** 2)
        distances.append(dist)
    fig, ax = plt.subplots()
    kpl.plot_points(ax, distances, avg_snr)
    ax.set_xlabel("Distance from center frequencies (MHz)")
    ax.set_ylabel("SNR")


def main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1]):
    ### Some initial setup
    num_steps = 1
    # uwave_ind_list = [0]
    # uwave_ind_list = [1]
    # uwave_ind_list = [0, 1]

    seq_file = "scc_snr_check.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
        ]

        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data

    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        charge_prep_fn=None,
        num_exps=2,
    )

    ### Report results and cleanup

    try:
        figs = process_and_plot(data)
    except Exception:
        print(traceback.format_exc())
        figs = None

    timestamp = dm.get_time_stamp()

    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)
    if figs is not None:
        num_figs = len(figs)
        for ind in range(num_figs):
            file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + f"-{ind}")
            dm.save_figure(figs[ind], file_path)

    tb.reset_cfm()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # # data = dm.get_raw_data(file_id=1730229940686)  # -8
    # # data = dm.get_raw_data(file_id=1730258128607)  # +0
    data = dm.get_raw_data(file_id=1730275767986)  # +8
    # figs = process_and_plot(data)
    # kpl.show(block=True)
    nv_list: list[NVSig] = data["nv_list"]
    nv_nums = [widefield.get_nv_num(nv) for nv in nv_list]
    num_nvs = len(nv_list)

    # fmt: off
    snr_lists = [
        [0.19211641111808578, 0.16861739214935872, 0.181115998643568, 0.1520350570328786, 0.08399650033935889, 0.1875575862456041, 0.09713658413721715, 0.09046049962392728, 0.08550917745701249, 0.12151842185669197, 0.0650640224802605, 0.18622382655858752, 0.14609498318343891, 0.045357623713732684, 0.10084277515246842, 0.13327826689917718, 0.16035895400879854, 0.059380635107353544, 0.09586631013626867, 0.15566039234398982, 0.04721313063587832, 0.13121901729644575, 0.07785422305639117, 0.061038640037846, 0.17172528479137691, 0.10077929125158978, 0.076974449866997, 0.022348756224880917, 0.11440638279056237, 0.03777617232306796, 0.08701988977432568, 0.10733180032802017, 0.1509368739468945, 0.15562469268075246, 0.11998670446943911, 0.07114678279037077, 0.07615861663269727, 0.009285287603357974, 0.13575085684992746, 0.1688665029099301, 0.17986258965588714, 0.10419718471109403, 0.14996032889388725, 0.11134616072184463, 0.10281917594232552, 0.08856905483189406, 0.13363722297594294, 0.11805851181284682, 0.0419659444565974, 0.08602787706177972, 0.17588450502366126, 0.0905005024078852, 0.19229975813489236, 0.15476799556612225, 0.07759670790943285, 0.04199688498231961, 0.10075552853873009, 0.04496767209800426, 0.07210084054715708, 0.1293264761485179, 0.10657129626833733, 0.1429942018232196, 0.11034707018054077, 0.14629621781753122, 0.12330568853989968, 0.079350459637556, 0.021569402015964103, 0.07862633516617588, 0.03565357846199471, 0.13552140902986692, 0.1591719708395756, 0.08454469037733979, 0.14321250090063575, 0.15711059901329022, 0.08378811409330371, 0.10637347308217311, 0.06780391636228258, 0.15154107680516457, 0.14068473659395483, 0.10582375151959027, 0.09115332228686027, 0.03286954567779138, 0.15450775871878064, 0.1453451104532476, 0.16020110143536645, 0.08077748418914514, 0.05379962322924675, 0.08853987791749103, 0.14972909701103415, 0.11974139170272954, 0.07891376160908593, 0.13449749556286536, 0.09891218277682809, 0.16172295578456708, 0.12398016814926027, 0.10249840507916319, 0.13985736195171125, 0.08978147501653877, 0.14640978339294614, 0.1263904542216376, 0.15881797702836045, 0.10431895813548933, 0.13206279123836026],
        [0.1712258892803559, 0.1624577591196261, 0.1672853528005329, 0.18130864578000225, 0.08151132335882814, 0.23070662735957168, 0.09948818507391248, 0.05168183622857892, 0.11414730548603988, 0.0879829825686134, 0.05277376820674089, 0.17157599199545853, 0.14454844820502624, 0.04530775620497695, 0.0622617514760073, 0.17083236013419256, 0.20296234946041378, 0.04344145028340419, 0.11184347971846689, 0.14923922769812242, 0.06905687299745755, 0.11992127437627692, 0.10585010903229411, 0.04504186771450093, 0.14132298206035898, 0.1198991291181971, 0.0679351579969144, 0.040235766242812394, 0.13641229218908527, 0.022266008006673565, 0.10755631160765623, 0.12742932813270202, 0.13192178319633258, 0.17129337307248826, 0.12717505972885354, 0.07470744131895841, 0.09396037438417677, 0.019675877910943645, 0.11083881623712646, 0.1768108632129926, 0.181056288620199, 0.08994968180658966, 0.15770063311609706, 0.1136354621956908, 0.10600862745733593, 0.08577092247389421, 0.12323494770063557, 0.13695938738524954, 0.07734441104405457, 0.06954411772848372, 0.18677770847001765, 0.09408890884265529, 0.17397251858293383, 0.1655293169366154, 0.06666179309853204, 0.07299776459776967, 0.12353998568624294, 0.030074521535991525, 0.08016763344027715, 0.10974084790870965, 0.1105870080215742, 0.16456067284156484, 0.09736250313152722, 0.15100114487192, 0.11514618985306484, 0.11161691387652413, 0.05261274149412361, 0.09820622992975046, 0.04558947327375281, 0.13626664515023798, 0.16282047515916528, 0.1073862871555339, 0.15576982985089, 0.1295181018486432, 0.07395931558837818, 0.09795922236577399, 0.07155660803002706, 0.16045206518596888, 0.14241036671663812, 0.11659363505306168, 0.08851897313318076, 0.030686450927884735, 0.13201452699120267, 0.16886335983487313, 0.1604954209308296, 0.12387753270480709, 0.06476997466908932, 0.11063025641433875, 0.14520948231936645, 0.13443493181777036, 0.08823190167819671, 0.1328199557484875, 0.09646631832916977, 0.1603888659526596, 0.17869986790878215, 0.1449745164317496, 0.13002197182984418, 0.1217366131897787, 0.16335199787311622, 0.1722490720198637, 0.1629820212551045, 0.10023235054821032, 0.15361906738271555],
        [0.14938546511554682, 0.13461883173332975, 0.14727514617635745, 0.17817572683216587, 0.07900948697155982, 0.24803291124413365, 0.12860050545654836, 0.06724088021076634, 0.10533094854620804, 0.09813728220125757, 0.057829628655067064, 0.16194482545264388, 0.1428193722331892, 0.03316246080536048, 0.05863208824465754, 0.18023348269234757, 0.1986716158859659, 0.04549342264618348, 0.09940966816301075, 0.12243320513080827, 0.046003391788622554, 0.16217667512963024, 0.11072228228871961, 0.04087342087142092, 0.15664453789287844, 0.12380029779356762, 0.05222949086619773, 0.042131859706496945, 0.1381918087842862, 0.03281803171213243, 0.15392122824643636, 0.10667778953812758, 0.09042253372462865, 0.19808198413436315, 0.08419917665052616, 0.07911936987816949, 0.10408384270805149, 0.013151994350898059, 0.09611746601099753, 0.1981869521213715, 0.16610978574279267, 0.06922723639103302, 0.14137781874312097, 0.09602644916528194, 0.12376192651501593, 0.06600750961393911, 0.13320651294211033, 0.07626207450920326, 0.07970716967112174, 0.09715549922814797, 0.20192016519875003, 0.10346441739505248, 0.13469651157958884, 0.14283648980280342, 0.09665032390112024, 0.07004756472194236, 0.11824644409730567, 0.034425070572285825, 0.08325150988882819, 0.09705422457302827, 0.1282428476209251, 0.18665668525035906, 0.06026476706014913, 0.13052432102611497, 0.10363629530803033, 0.13090966651722657, 0.07064982778201288, 0.11260604008007191, 0.05717802657687122, 0.14285307356922144, 0.14181889125473726, 0.10402498485790028, 0.16204046405649628, 0.14202369506881019, 0.10716986418580085, 0.08843326206912308, 0.06911829228856006, 0.12414857919532761, 0.10999111523588061, 0.09198294381735571, 0.09985741474311692, 0.05117080654438622, 0.129080413443608, 0.1340920530630369, 0.12160066648348711, 0.12137006415674169, 0.07052716183859986, 0.11440738828739427, 0.09611886581406975, 0.15062872565754887, 0.08187987913422266, 0.11462279696022046, 0.0628975056603484, 0.12733695532740655, 0.17189466499206144, 0.13319301127427846, 0.10537511991363901, 0.11553893222855895, 0.19635583748861965, 0.17208279986957337, 0.11776801428296298, 0.07517968877206165, 0.15969619606873992],
    ]
    # fmt: on
    snr_lists = np.array(snr_lists)
    orientation_data = dm.get_raw_data(file_id=1723161184641)
    orientation_a_nums = orientation_data["orientation_indices"]["0.041"]["nv_indices"]
    orientation_b_nums = orientation_data["orientation_indices"]["0.147"]["nv_indices"]
    orientation_ab_nums = orientation_a_nums + orientation_b_nums
    orientation_a_inds = [nv_nums[ind] in orientation_a_nums for ind in range(num_nvs)]
    orientation_b_inds = np.logical_not(orientation_a_inds)

    mean_snrs = np.mean(snr_lists, axis=1)
    mean_snrs_a = np.mean(snr_lists[:, orientation_a_inds], axis=1)
    mean_snrs_b = np.mean(snr_lists[:, orientation_b_inds], axis=1)

    fig, ax = plt.subplots()
    taus = [56, 64, 72]
    kpl.plot_points(ax, taus, mean_snrs, label="All NVs")
    kpl.plot_points(ax, taus, mean_snrs_a, label="Orientation A")
    kpl.plot_points(ax, taus, mean_snrs_b, label="Orientation B")
    ax.legend()
    kpl.show(block=True)
