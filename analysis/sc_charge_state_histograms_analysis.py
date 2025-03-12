# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference, while fitting a bimodal distribution to NV charge states.

Created on Fall 2024

@author: saroj chand
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


# Update the plot_histograms function for better visualization
def plot_histograms(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    no_text=None,
    ax=None,
    density=False,
    nv_index=None,
):
    """Plot histograms for signal and reference counts with enhanced visualization."""
    sns.set_theme(style="whitegrid")  # Use a Seaborn theme for improved aesthetics

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = sns.color_palette("husl", 2)  # Use Seaborn color palette
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()  # Larger figure size for clarity
    else:
        fig = None

    if not no_title:
        ax.set_title(
            f"Charge Prep Histogram ({num_reps} reps)", fontsize=14, weight="bold"
        )

    ax.set_xlabel("Integrated Counts", fontsize=12)
    ax.set_ylabel("Probability" if density else "Occurrences", fontsize=12)

    for ind, counts_list in enumerate(counts_lists):
        sns.histplot(
            counts_list,
            kde=False,
            stat="density" if density else "count",
            bins=50,
            ax=ax,
            label=labels[ind],
            color=colors[ind],
            alpha=0.7,
        )

    ax.legend(title="Pulse Type", fontsize=10, loc="upper right", title_fontsize=12)
    plt.show(block=True)
    if fig is not None:
        return fig


# Update scatter plot aesthetics
def scatter_plot(x_data, y_data, xlabel, ylabel, title):
    """Create a scatter plot with purple markers and transparent filling."""
    plt.figure()
    plt.scatter(
        x_data,
        y_data,
        edgecolors="black",  #  circle outlines
        facecolors="blue",  #  fill color
        alpha=0.6,  # Transparency for the filling
        s=60,  # Marker size
        linewidth=0.6,  # Outline thickness
    )
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Update the image plotting function for improved visuals
def plot_images(img_arrays, readout_laser, readout_ms, title_suffixes):
    """Plot images with improved Seaborn style."""
    sns.set_theme(style="darkgrid")  # Change to a darker grid style for images
    img_figs = []

    for ind, img_array in enumerate(img_arrays):
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        sns.heatmap(
            img_array,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Photons"},
            annot=False,
        )
        ax.set_title(
            f"{readout_laser}, {readout_ms:.2f} ms, {title_suffix}", fontsize=14
        )
        img_figs.append(fig)

    return img_figs


def process_and_plot(
    raw_data, do_plot_histograms=False, prob_dist: ProbDist = ProbDist.COMPOUND_POISSON
):
    ### Setup
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    ### Histograms and thresholding
    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    hist_figs = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=True
        )
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=False, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        prep_fidelity_list.append(prep_fidelity)

        # Plot histograms
        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            if fig is not None:
                hist_figs.append(fig)

    # Report averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    med_readout_fidelity = np.nanmedian(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    med_prep_fidelity = np.nanmedian(prep_fidelity_list)
    print(f"Average Readout Fidelity: {avg_readout_fidelity:.3f}")
    print(f"Median Readout Fidelity: {med_readout_fidelity:.3f}")
    print(f"Average NV- Preparation Fidelity: {avg_prep_fidelity:.3f}")
    print(f"Median NV- Preparation Fidelity: {med_prep_fidelity:.3f}")
    selected_nv_indices = [
        ind
        for ind in range(num_nvs)
        if prep_fidelity_list[ind] > 0.4 and readout_fidelity_list[ind] > 0.7
    ]
    print(f"Selected NVs: {selected_nv_indices}")
    print(f"len(selected_nv_indices): {len(selected_nv_indices)}")
    # manual removal of indices
    print(num_nvs)
    print(readout_fidelity_list[5])
    indices_to_remove = [18, 35, 56]
    readout_fidelity_list = [
        readout_fidelity_list[idx]
        for idx in range(num_nvs)
        if idx not in indices_to_remove
    ]
    prep_fidelity_list = [
        prep_fidelity_list[idx]
        for idx in range(num_nvs)
        if idx not in indices_to_remove
    ]
    # Scatter plot: Readout fidelity vs Prep fidelity
    scatter_plot(
        readout_fidelity_list,
        prep_fidelity_list,
        xlabel="Readout Fidelity",
        ylabel="NV- Preparation Fidelity",
        title="Readout vs Prep Fidelity",
    )

    # Scatter plot: Distance from center vs Prep fidelity
    coords_key = "laser_INTE_520_aod"
    # distances = [
    #     np.sqrt(
    #         (110 - pos.get_nv_coords(nv, coords_key, drift_adjust=False)[0]) ** 2
    #         + (110 - pos.get_nv_coords(nv, coords_key, drift_adjust=False)[1]) ** 2
    #     )
    #     for nv in nv_list
    # ]
    # scatter_plot(
    #     distances,
    #     prep_fidelity_list,
    #     xlabel="Distance from Center (MHz)",
    #     ylabel="NV- Preparation Fidelity",
    #     title="Prep Fidelity vs Distance",
    # )

    # Image plotting
    if "img_arrays" not in raw_data:
        return

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["physical_name"]
    readout_ms = laser_dict["duration"] / 10**6

    img_arrays = raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["Signal", "Reference", "Difference"]

    img_figs = plot_images(
        img_arrays_to_save, readout_laser, readout_ms, title_suffixes
    )

    return img_arrays_to_save, img_figs, hist_figs


def fidelities_test():
    # fmt:off
    prep_fidelity_list_60ms_108uW= [0.5737943936055317, 0.4048672365635224, 0.6080205649481236, 0.5901836691023351, 0.5079887528316527, 0.5542955309498643, 0.5578104702376074, 0.5607242560609407, 0.6101713394245407, 0.4521288875225675, 0.5954985929194911, 0.5904271963564154, 0.42021314291604805, 0.5015917866096284, 0.5458607214191309, 0.6296413127445364, 0.5536980092414325, 0.5343584997332287, 0.2623965403560856, 0.39601266680201974, 0.5797863342583169, 0.5114759301554543, 0.5256116514940693, 0.5229066445681925, 0.450047285630473, 0.572292644907745, 0.5976582466152901, 0.5130485311646913, 0.5273861929619351, 0.34757531504374917, 0.5502104842543301, 0.4456862368359711, 0.553632611954382, 0.5402529826605105, 0.6861548933575234, 0.08481508719643627, 0.5244637501384432, 0.4377843414463414, 0.514523254601241, 0.4463240300282022, 0.6265427820763659, 0.6709437979974808, 0.32283005266989984, 0.542952832527758, 0.3010986804653031, 0.41343134727712183, 0.46108182159144784, 0.4461200498401633, 0.4815122744231428, 0.4048638865578642, 0.39410092855983847, 0.5558524238874132, 0.5481261337132645, 0.46866745249061814, 0.37843232260554116, 0.5299810910420883, 0.19781536289170487, 0.37233503567753845, 0.49462985746833543, 0.23251333066998614, 0.5387586880756504, 0.5635408918875476, 0.5068152407584695, 0.4040954496153156, 0.5061587016841012, 0.6562393240442146, 0.4086604834788129, 0.43003181887758846, 0.37087811354313893, 0.7028345966277282, 0.41766675753176086, 0.5192372892236826, 0.3638778018815393, 0.5121756334853828, 0.33814896037380104, 0.2946791773710504, 0.42927627713554983, 0.6797212416873737, 0.4860730975938444, 0.5372719838654767, 0.9136860045527515, 0.7532491840327614, 0.46398577499903937, 0.3980369433758283, 0.33743195063301046, 0.44831138346605093, 0.3102021141595366, 0.3356355710216572, 0.43628940987876264, 0.4642684087176008, 0.6654038487506226, 0.6095250355479236, 0.44056160538355205, 0.68662405028677, 0.5341482601817285, 0.5101878164785933, 0.41697132477114074, 0.34494402408577085, 0.5136756179201557, 0.6292990118140847, 0.5446501838148083, 0.41612559713367026, 0.5354019482513129, 0.45493933336687364, 0.5981129835320359, 0.5400728293324901, 0.39816427949084576, 0.494029907168139, 0.6112603165864454, 0.5182677809806657, 0.4848065977935925, 0.14234973510798243, 0.6431503485437883, 0.45542115658568083, 0.4337886439181037, 0.6543740533862875, 0.4083522066724644, 0.6395353381656275, 0.5176784300460741, 0.1889469143908139, 0.5785336103884486, 0.539572530438262, 0.24249484005721433, 0.6082951415314178, 0.5665951184897027, 0.5854074923638359, 0.639397435972253, 0.4064534537970962, 0.3276334408957372, 0.4658784310617522, 0.5012261916505485]
    prep_fidelity_list_30ms_216uW= [0.5638474175472732, 0.36349131210886587, 0.5791551632828735, 0.6528442651099204, 0.5303721153884301, 0.5409407143596854, 0.5745768648527401, 0.6077783500190652, 0.5246045372873727, 0.5168047457594835, 0.5861055405384432, 0.5793594611625167, 0.4900083735927667, 0.562650093389587, 0.4728844592023078, 0.6321176688623763, 0.5413734215509871, 0.6031764764997616, 0.2795302355168564, 0.3875942777722484, 0.5398655502322307, 0.5477388855646631, 0.5401992649341439, 0.4709086465851813, 0.47117575325669836, 0.5314511462702858, 0.4832494435274247, 0.5484333863390911, 0.4996387006384069, 0.3522217669972598, 0.5776339012850562, 0.47881123037064655, 0.5747172416000627, 0.5781913747281364, 0.6688520154461397, 0.31910743464068003, 0.5291543433816581, 0.45127300879893184, 0.5102854508363602, 0.3861200470357522, 0.6017572566025016, 0.6613487465645347, 0.3354164317761189, 0.10206978569803182, 0.31169249216052297, 0.42994471215369945, 0.48309854101896976, 0.45701585845296755, 0.5174881055409326, 0.3798547771732381, 0.5788532765179296, 0.5467011754955429, 0.590486944272351, 0.4633895407250497, 0.3707718359860904, 0.47284049765281244, 0.3863581077805964, 0.49048645756960285, 0.5248704768302369, 0.27866278295249114, 0.5147058314946964, 0.3751307106697379, 0.5028935298541524, 0.4561687665895736, 0.5138109531075632, 0.6710532828949249, 0.3718360656389781, 0.4427007007858096, 0.4162320473682921, 0.4033121178031094, 0.4316528206368452, 0.4985804119456828, 0.39983715570491785, 0.37668700186131165, 0.3253412350582173, 0.24646224297733577, 0.4200168927195167, 0.6152508206308599, 0.5021880498512077, 0.4511580024395747, 0.6550351872114153, 0.7228427148252701, 0.4874376244754436, 0.36954226328589856, 0.3954083640216176, 0.397513518578562, 0.3495813652519455, 0.3798377046989202, 0.4609950157836614, 0.42375624611370666, 0.619862505424166, 0.5448749859859099, 0.45012264004125835, 0.5320408990987422, 0.4582959541366848, 0.6401410236912088, 0.47385989516497473, 0.43561617736472913, 0.5042458419538614, 0.5320991392802079, 0.5382859908499625, 0.41371361022542585, 0.5409681486386291, 0.61976681679686, 0.7024612610597943, 0.5156647132389345, 0.40183888979730853, 0.47223555994759514, 0.5406816831228878, 0.47741273518033334, 0.5341371440857738, 0.11076432595064001, 0.6404810776234364, 0.5045702464947608, 0.3916854533367712, 0.5136329736223781, 0.3735423158892569, 0.5931734453193189, 0.5352568814616536, 0.20787868840025325, 0.6322892391054394, 0.476301723789616, 0.20059216240264766, 0.416350609268926, 0.5981697835278088, 0.5106304664129395, 0.2867626397280578, 0.4836052678499858, 0.3186930576648548, 0.4091316331845374, 0.4681547895557382]
    # fmt:on
    import scipy.stats as stats

    # Compute statistics
    def compute_stats(fidelity_list):
        mean = np.mean(fidelity_list)
        std_dev = np.std(fidelity_list)
        return mean, std_dev

    mean_60ms, std_60ms = compute_stats(prep_fidelity_list_60ms_108uW)
    mean_30ms, std_30ms = compute_stats(prep_fidelity_list_30ms_216uW)

    # Print statistics
    print(f"60ms 108uW: Mean = {mean_60ms:.4f}, Std Dev = {std_60ms:.4f}")
    print(f"30ms 216uW: Mean = {mean_30ms:.4f}, Std Dev = {std_30ms:.4f}")

    # Scatter plot comparison
    plt.figure(figsize=(8, 6))
    plt.scatter(
        range(len(prep_fidelity_list_60ms_108uW)),
        prep_fidelity_list_60ms_108uW,
        label="60ms 108uW",
        alpha=0.7,
    )
    plt.scatter(
        range(len(prep_fidelity_list_30ms_216uW)),
        prep_fidelity_list_30ms_216uW,
        label="30ms 216uW",
        alpha=0.7,
    )
    plt.xlabel("NV Index")
    plt.ylabel("Preparation Fidelity")
    plt.legend()
    plt.title("Preparation Fidelity Comparison")
    plt.show()

    # Histogram comparison
    plt.figure(figsize=(8, 6))
    plt.hist(prep_fidelity_list_60ms_108uW, bins=20, alpha=0.5, label="60ms 108uW")
    plt.hist(prep_fidelity_list_30ms_216uW, bins=20, alpha=0.5, label="30ms 216uW")
    plt.xlabel("Preparation Fidelity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Preparation Fidelity")
    plt.show()

    # Correlation
    correlation, p_value = stats.pearsonr(
        prep_fidelity_list_60ms_108uW, prep_fidelity_list_30ms_216uW
    )
    print(
        f"Correlation between 60ms 108uW and 30ms 216uW: {correlation:.4f} (p-value: {p_value:.4f})"
    )


if __name__ == "__main__":
    kpl.init_kplotlib()
    # Process and plot function and Set Seaborn theme globally for consistent styling
    # sns.set_theme(style="whitegrid")
    # data = dm.get_raw_data(file_id=1754374316674, load_npz=False)
    # data = dm.get_raw_data(file_id=1766803842180, load_npz=False)  # 50ms readout
    # data = dm.get_raw_data(file_id=1766834596476, load_npz=False)  # 100ms readout
    # data = dm.get_raw_data(file_id=1770828500425, load_npz=False)  # 60ms readout
    # data = dm.get_raw_data(file_id=1778189406841, load_npz=False)  # 60ms readout 61 NVs
    # data = dm.get_raw_data(file_id=1782616297820, load_npz=False)  # 60ms readout 66 NVs
    # data = dm.get_raw_data(file_id=1791781168217, load_npz=False)  # 60ms readout 66 NVs

    # rubin
    # data = dm.get_raw_data(
    #     file_id=1794036299375, load_npz=False
    # )  # 60ms readout 140 NVs
    # file_name = dm.get_file_name(file_id=1766803842180)
    # file_name = dm.get_file_name(file_id=1766803842180)
    # data = dm.get_raw_data(file_id=1794714155833, load_npz=False)
    data = dm.get_raw_data(file_id=1796486502363, load_npz=False)
    # data = dm.get_raw_data(file_id=1796486502363, load_npz=False)
    data = dm.get_raw_data(file_id=1796486502363, load_npz=False)

    # print(file_name)
    # process_and_plot(data, do_plot_histograms=True)
    fidelities_test()
    kpl.show(block=True)
