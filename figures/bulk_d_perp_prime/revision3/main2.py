# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:45 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar as scale_bar
import json
import matplotlib.font_manager
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
ms = 7
lw = 1.75


# %% Functions


def blurb(file_names, nv_data):
    
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # fig.set_tight_layout(True)
    lower_ax = ax
    
    tick_pos = []
    tick_labels = []
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        if ind == 0:
            label_gamma = r'$\gamma$'
            label_omega = r'$\Omega$'
        else:
            label_gamma = None
            label_omega = None
            
        gamma_plot = ax.errorbar(ind, nv['gamma']*1000, yerr=nv['gamma_err']*1000, 
                    label=label_gamma, marker='o',
                    color='#993399', markerfacecolor='#CC99CC',
                    linestyle='None', ms=ms, lw=lw)
        
        omega_plot = ax.errorbar(ind, nv['omega']*1000, yerr=nv['omega_err']*1000, 
                    label=label_omega, marker='^',
                    color='#FF9933', markerfacecolor='#FFCC33',
                    linestyle='None', ms=ms, lw=lw)
        
        tick_pos.append(ind)
        tick_labels.append(nv['name'])
        
    ax.set_ylabel('Relaxation rate (s$^{-1}$)')
    ax.set_ylim(45, 135)
    ax.set_xlabel(None)
    ax.set_xlim(-0.35, 4.4)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc='upper right')
    # r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
    #                               visible=False)
    # legend = ax.legend([r, omega_plot, r, r, gamma_plot, r],
    #                     ['', r'$\Omega$', '', '', r'$\gamma$', ''],
    #                     bbox_to_anchor=(0., 1.02, 1., .1), loc='lower left',
    #                     ncol=6, mode='expand', borderaxespad=0.0, handlelength=0.5)
    
    
    # %% perp_B = 0 ratio
    
    plt.rcParams.update({'font.size': 12})

    # Create inset of width 30% and height 40% of the parent axes' bounding box
    # at the lower left corner (loc=3)
    ax = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(0.695, 0.10, 0.3, 0.30), bbox_transform=ax.transAxes
                    # loc='lower right',
                    )
    
    # ax = fig.add_subplot(gs[2:5, 3:6])
    
    tick_pos = []
    tick_labels = []
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        ax.errorbar(ind, nv['ratio'], yerr=nv['ratio_err'], 
                    label=label_gamma, marker='o',
                    color='#EF2424', markerfacecolor='#FB9898',
                    linestyle='None', ms=ms, lw=lw)
        
        tick_pos.append(ind)
        tick_labels.append(nv['name'])
        
    ax.set_ylabel(r'$\gamma / \Omega$')
    ax.set_yticks([1.8,2.0,2.2])
    ax.set_xlabel(None)
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    
    plt.rcParams.update({'font.size': 15})
    # fig.tight_layout()
    fig.tight_layout(pad=0.3, h_pad=1.0, w_pad=0.3, rect=[0,0,1,1])
    

# %% Main


def main(file_names, nv_data):
    """
    2 x 2 figure. Top left ax blank for level structure, next 3 for sample
    scans. file_names should be a list with 3 paths to the appropriate
    raw data files.
    """
    
    figwidth = 6.75
    fig = plt.figure(figsize=(figwidth, 5.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.4, 0.5])  
    
    # %% Sample scans
    
    centers = [[96, 31], [100, 100], [50,50]]
    fig_labels = [r'(a)', r'(b)', r'(c)']
    sample_labels = ['Sample A', 'Sample B', 'Sample C']
    
    for ind in range(3):
        
        ax = fig.add_subplot(gs[0, ind])
        name = file_names[ind]
        
        # This next bit is jacked from image_sample.reformat_plot 2/29/2020
        with open(name) as file:

            # Load the data from the file
            data = json.load(file)

            # Build the image array from the data
            # Not sure why we're doing it this way...
            img_array = []
            try:
                file_img_array = data['img_array']
            except:
                file_img_array = data['imgArray']
            for line in file_img_array:
                img_array.append(line)

            # Get the readout in s
            readout = float(data['readout']) / 10**9

            try:
                xScanRange = data['x_range']
                yScanRange = data['y_range']
            except:
                xScanRange = data['xScanRange']
                yScanRange = data['yScanRange']
            
        kcps_array = (numpy.array(img_array) / 1000) / readout
        
        # Scaling
        scale = 35  # galvo scaling in microns / volt
        num_steps = kcps_array.shape[0]
        v_resolution = xScanRange / num_steps  # resolution in volts / pixel
        resolution = v_resolution * scale  # resolution in microns / pixel
        px_per_micron = 1/resolution
        
        # Plot! 5 um out from center in any direction
        center = centers[ind]
        clip_range = 5 * px_per_micron 
        x_clip = [center[0] - clip_range, center[0] + clip_range]
        x_clip = [int(el) for el in x_clip]
        y_clip = [center[1] - clip_range, center[1] + clip_range]
        y_clip = [int(el) for el in y_clip]
        # print((x_clip[1] - x_clip[0]) * v_resolution)
        clip_array = kcps_array[x_clip[0]: x_clip[1], 
                                y_clip[0]: y_clip[1]]
        img = ax.imshow(clip_array, cmap='inferno', interpolation='none')
        ax.set_axis_off()
        
        # Scale bar
        trans = ax.transData
        bar_text = r'2 $\upmu$m '  # Insufficient left padding...
        bar = scale_bar(trans, 2*px_per_micron, bar_text, 'upper right',
                        size_vertical=int(num_steps/100),
                        pad=0.25, borderpad=0.5, sep=4.0,
                        # frameon=False, color='white',
                        )
        ax.add_artist(bar)
        
        # Labels
        fig_label = fig_labels[ind]
        ax.text(0.035, 0.88, fig_label,
                transform=ax.transAxes, color='white', fontsize=18)
        sample_label = sample_labels[ind]
        ax.text(0.035, 0.07, sample_label,
                transform=ax.transAxes, color='white')
        # cbar = plt.colorbar(img)
        
    # %% perp_B = 0 gamma and omega
        
    ax = fig.add_subplot(gs[1, 0:3])
    lower_ax = ax
    
    tick_pos = []
    tick_labels = []
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        if ind == 0:
            label_gamma = r'$\gamma$'
            label_omega = r'$\Omega$'
        else:
            label_gamma = None
            label_omega = None
            
        gamma_plot = ax.errorbar(ind, nv['gamma']*1000, yerr=nv['gamma_err']*1000, 
                    label=label_gamma, marker='o',
                    color='#993399', markerfacecolor='#CC99CC',
                    linestyle='None', ms=ms, lw=lw)
        
        omega_plot = ax.errorbar(ind, nv['omega']*1000, yerr=nv['omega_err']*1000, 
                    label=label_omega, marker='^',
                    color='#FF9933', markerfacecolor='#FFCC33',
                    linestyle='None', ms=ms, lw=lw)
        
        tick_pos.append(ind)
        tick_labels.append(nv['name'])
        
    ax.set_ylabel('Relaxation rate (s$^{-1}$)')
    ax.set_xlabel(None)
    ax.set_xlim(-0.25, 3.5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc='upper right')
    # r = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
    #                               visible=False)
    # legend = ax.legend([r, omega_plot, r, r, gamma_plot, r],
    #                     ['', r'$\Omega$', '', '', r'$\gamma$', ''],
    #                     bbox_to_anchor=(0., 1.02, 1., .1), loc='lower left',
    #                     ncol=6, mode='expand', borderaxespad=0.0, handlelength=0.5)
    
    # Fig label
    fig_label = fig_labels[ind]
    ax.text(-0.143, 0.97, '(d)', transform=ax.transAxes,
            color='black', fontsize=18)
    
    # %% perp_B = 0 ratio
    
    plt.rcParams.update({'font.size': 12})

    # Create inset of width 30% and height 40% of the parent axes' bounding box
    # at the lower left corner (loc=3)
    ax = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(0.770, 0.18, 0.23, 0.40), bbox_transform=ax.transAxes
                    # loc='lower right',
                    )
    
    # ax = fig.add_subplot(gs[2:5, 3:6])
    
    tick_pos = []
    tick_labels = []
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        ax.errorbar(ind, nv['ratio'], yerr=nv['ratio_err'], 
                    label=label_gamma, marker='o',
                    color='#EF2424', markerfacecolor='#FB9898',
                    linestyle='None', ms=ms, lw=lw)
        
        tick_pos.append(ind)
        tick_labels.append(nv['name'])
        
    ax.set_ylabel(r'$\gamma / \Omega$')
    ax.set_yticks([1.8,2.0,2.2])
    ax.set_xlabel(None)
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    
    plt.rcParams.update({'font.size': 15})
    
    # %% Wrap up
        
    shift = 0.103
    # shift = 0.125
    gs.tight_layout(fig, pad=0.3, h_pad=1.0, w_pad=0.3, rect=[-shift,0,1,1])
    pos = lower_ax.get_position().bounds
    shift += 0.02
    lower_ax.set_position([pos[0]+shift, pos[1], pos[2]-shift, pos[3]])
    


# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)

    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample/'
    file_names = ['2019_07/2019-07-23_17-39-48_johnson1.txt',
                  '2019_10/2019-10-02-15_12_01-goeppert_mayer-nv_search.txt',
                  '2019_04/2019-04-15_16-42-08_Hopper.txt']
    file_names = [path+name for name in file_names]
    
    # NV data in three samples at B perp ~ 0
    nv_data = [
        {
            'name': 'NVA3',  # perp_B = 0.5536
            'gamma': 0.114,
            'gamma_err': 0.01,
            'omega': 0.059,
            'omega_err': 0.004,
            'ratio': 1.9322,
            'ratio_err': 0.2142,
            
        },
        {
            'name': 'NVB1',  # Weighted averages for both perp_B = 0 runs
            'gamma': 0.117,
            'gamma_err': 0.007071068,
            'omega': 0.057731707,
            'omega_err': 0.003123475,
            'ratio': 2.000849784,
            'ratio_err': 0.162765942,
        },
        {
            'name': 'NVC',  # perp_B = 0.0
            'gamma': 0.12,
            'gamma_err': 0.011,
            'omega': 0.061,
            'omega_err': 0.004,
            'ratio': 1.9672,
            'ratio_err': 0.2217,
        }
    ]
    
    # main(file_names, nv_data)
    blurb(file_names, nv_data)
