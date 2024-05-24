# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on April 9th, 2019

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np

def post_process_saved_data(file_path, bin_width=50):
    # Load saved data
    saved_data = np.load(file_path, allow_pickle=True)

    # Print the keys in the loaded data
    #print("Keys in the loaded data:", saved_data.keys())

    sig_img_array_list = saved_data["sig_img_array_list"]
    ref_img_array_list = saved_data["ref_img_array_list"]

    bg_coords_outside = (90, 90)  # Background coordinates outside aperture
    bg_coords_inside = (200,350)  # Background coordinates inside aperture
    nv_coords = (253,321)  # NV coordinates
    #nv_coords = (321,253)  # NV coordinates
    pixel_range = 6  # Number of pixels around the coordinates

    bg_counts_outside_sig = []
    bg_counts_outside_ref = []
    bg_counts_inside_sig = []
    bg_counts_inside_ref = []
    nv_counts_sig = []
    nv_counts_ref = []

    all_sig_img_array = np.zeros_like(sig_img_array_list[0])
    all_ref_img_array = np.zeros_like(ref_img_array_list[0])

    for sig_img_array, ref_img_array in zip(sig_img_array_list, ref_img_array_list):
        # Background counts outside aperture
        bg_outside_sig = np.sum(sig_img_array[bg_coords_outside[0]-pixel_range:bg_coords_outside[0]+pixel_range, bg_coords_outside[1]-pixel_range:bg_coords_outside[1]+pixel_range])
        bg_outside_ref = np.sum(ref_img_array[bg_coords_outside[0]-pixel_range:bg_coords_outside[0]+pixel_range, bg_coords_outside[1]-pixel_range:bg_coords_outside[1]+pixel_range])
        bg_counts_outside_sig.append(bg_outside_sig)
        bg_counts_outside_ref.append(bg_outside_ref)

        # Background counts inside aperture
        bg_inside_sig = np.sum(sig_img_array[bg_coords_inside[0]-pixel_range:bg_coords_inside[0]+pixel_range, bg_coords_inside[1]-pixel_range:bg_coords_inside[1]+pixel_range])
        #plt.imshow(sig_img_array[bg_coords_inside[0]-pixel_range:bg_coords_inside[0]+pixel_range, bg_coords_inside[1]-pixel_range:bg_coords_inside[1]+pixel_range])
        #plt.show(block=True)
        bg_inside_ref = np.sum(ref_img_array[bg_coords_inside[0]-pixel_range:bg_coords_inside[0]+pixel_range, bg_coords_inside[1]-pixel_range:bg_coords_inside[1]+pixel_range])
        bg_counts_inside_sig.append(bg_inside_sig)
        bg_counts_inside_ref.append(bg_inside_ref)

        # NV counts
        nv_sig = np.sum(sig_img_array[nv_coords[0]-pixel_range:nv_coords[0]+pixel_range, nv_coords[1]-pixel_range:nv_coords[1]+pixel_range])
        nv_ref = np.sum(ref_img_array[nv_coords[0]-pixel_range:nv_coords[0]+pixel_range, nv_coords[1]-pixel_range:nv_coords[1]+pixel_range])
        nv_counts_sig.append(nv_sig)
        nv_counts_ref.append(nv_ref)

        # Sum all sig_img_array and ref_img_array
        all_sig_img_array += sig_img_array
        all_ref_img_array += ref_img_array

    #plt.imshow(all_ref_img_array[nv_coords[0]-pixel_range:nv_coords[0]+pixel_range, nv_coords[1]-pixel_range:nv_coords[1]+pixel_range])
    #plt.show(block=True)

    # Create histograms in a three-panel figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].hist(bg_counts_outside_sig, bins= bin_width, alpha=0.5, label="Signal")
    axs[0].hist(bg_counts_outside_ref, bins= bin_width, alpha=0.5, label="Reference")
    axs[0].set_title("BG (Outside Aperture) Histogram")
    axs[0].set_ylabel("Events")
    axs[0].set_xlabel("Background Counts")
    axs[0].legend()

    axs[1].hist(bg_counts_inside_sig, bins= bin_width, alpha=0.5, label="Signal")
    axs[1].hist(bg_counts_inside_ref, bins= bin_width, alpha=0.5, label="Reference")
    axs[1].set_title("BG (Inside Aperture) Histogram")
    axs[1].set_ylabel("Events")
    axs[1].set_xlabel("Background Counts")
    axs[1].legend()

    axs[2].hist(nv_counts_sig, bins= bin_width, alpha=0.5, label="Signal")
    axs[2].hist(nv_counts_ref, bins= bin_width, alpha=0.5, label="Reference")
    axs[2].set_title("NV Counts Histogram")
    axs[2].set_ylabel("Events")
    axs[2].set_xlabel("NV Counts")
    axs[2].legend()

    plt.show(block=False)

    # Plot summed images
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    img1 = axs[0].imshow(all_sig_img_array, cmap='Spectral', interpolation='nearest')
    axs[0].set_title("Summed Signal Images")
    fig.colorbar(img1, ax=axs[0], orientation='vertical', label='Counts')

    img2 = axs[1].imshow(all_ref_img_array, cmap='Spectral', interpolation='nearest')
    axs[1].set_title("Summed Reference Images")
    fig.colorbar(img2, ax=axs[1], orientation='vertical', label='Counts')

    plt.show()

# Example usage
#file_path = "/Users/schand/Library/CloudStorage/GoogleDrive-schand@berkeley.edu/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample_diff/2023_11/2023_11_13-19_04_14-johnson-nv0_2023_11_09.npz"
#file_path = "/Users/schand/Library/CloudStorage/GoogleDrive-schand@berkeley.edu/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample_diff/2023_11/2023_11_13-17_38_44-johnson-nv0_2023_11_09.npz"
#file_path = "/Users/schand/Library/CloudStorage/GoogleDrive-schand@berkeley.edu/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample_diff/2023_11/2023_11_13-13_47_25-johnson-nv0_2023_11_09.npz"
file_path = "/Users/schand/Library/CloudStorage/GoogleDrive-schand@berkeley.edu/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample_diff/2023_11/2023_11_14-12_16_43-johnson-nv0_2023_11_09-diff.npz"
post_process_saved_data(file_path, bin_width=50)
