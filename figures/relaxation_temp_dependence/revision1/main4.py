# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:15 2019

@author: matth
"""


# %% Imports


import numpy
import matplotlib
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
ms = 7
lw = 1.75
    
# d_parallel = 0.35
# d_perp = 17
d_parallel = 0.35
d_perp = 17


# %% Functions


def intersection():
    
    meas_ratio = 1.973556424
    meas_err = 0.111886693
    
    factor = ((d_parallel - 2*d_perp)**2 + d_parallel**2) / 8
    d_perp_prime = numpy.sqrt(factor * meas_ratio)
    err = d_perp_prime * (1/2) * meas_err / meas_ratio
    
    print('d_perp_prime = {} +/- {}'.format(d_perp_prime, err))


def ratio(d_perp_prime):
    
    return (8 * d_perp_prime**2) / ((d_parallel - 2*d_perp)**2 + d_parallel**2)


def plot_ratio(ax, linspace_x):
    
    ax.plot(linspace_x, ratio(linspace_x), linewidth=lw, color='#0072B2')
    
    ax.set_xlabel(r"$d_{\perp}'$ (Hz cm/V)")
    ax.set_ylabel(r'$\gamma/\Omega$')
    ax.set_xticks([0,5,10,15, 20])
    
    lin_color = '#EF2424'
    fill_color = '#FBBFBF'  #'#FB9898'
    meas_ratio = 1.973556424
    meas_err = 0.111886693
    ax.plot(linspace_x, [meas_ratio]*1000, c=lin_color, linewidth=lw)
    ax.fill_between(linspace_x, meas_ratio - meas_err, meas_ratio + meas_err,
                    color=fill_color)
    
    
            

# %% Main


def main():

    # plt.rcParams.update({'font.size': 18})  # Increase font size
    # fig, axes_pack = plt.subplots(1,2, figsize=(10,5))
    fig = plt.figure(figsize=(6.75,6.75/2))
    gs = gridspec.GridSpec(1, 3)
    
    # source = 't1_double_quantum/paper_data/bulk_dq/'
    # path = source + folder
    
    # %% Level structure
    
    if False:
        # Add a new axes, make it invisible, steal its rect
        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        ax.text(-0.295, 1.05, '(a)', transform=ax.transAxes,
                color='black', fontsize=16)
        
        ax = plt.Axes(fig, [0.0, 0.51, 0.5, 0.43])
        ax.set_axis_off()
        fig.add_axes(ax)
        file = 'C:/Users/matth/Desktop/lab/bulk_dq_relaxation/figures_revision2/main1/level_structure.png'
        img = mpimg.imread(file)
        img_plot = ax.imshow(img)

    # %% d perp prime plot
    
    ax = fig.add_subplot(gs[0, 2])
    linspace_x = numpy.linspace(0, 20, 1000)
    plot_ratio(ax, linspace_x)
    ax.text(-0.35, 0.94, '(c)', transform=ax.transAxes,
            color='black', fontsize=16)
    
    # %% Dummy labels
    
    # fig.text(0.0, 0.94, '(a)', transform=fig.transFigure,
    #         color='black', fontsize=16)
    # fig.text(0.32, 0.94, '(b)', transform=fig.transFigure,
    #         color='black', fontsize=16)
    
    # %% Inset zoom
    
    # ax = inset_axes(ax, width="100%", height="100%",
    #                 bbox_to_anchor=(0.770, 0.18, 0.23, 0.40),
    #                 bbox_transform=ax.transAxes)
    # linspace_x = numpy.linspace(15, 20, 1000)
    # plot_ratio(ax, linspace_x)
    
    # %% Wrap up
    
    fig.tight_layout(pad=0.2)
    # fig.tight_layout()
    

# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 13})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)

    main()
    # print(ratio(17))
    # intersection()

