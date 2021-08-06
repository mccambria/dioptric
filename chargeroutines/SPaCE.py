# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:33:57 2020

A routine to take one NV to readout the charge state, after pulsing a laser
at a distance from this readout NV.


@author: agardill
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
from random import shuffle
import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats
from scipy.optimize import curve_fit


# Define the start time to track the time between optimzies
time_start = time.time() 

def inverse_func(x, c, a):
    return c + a*x**-1/2

def exp_decay(x, c, a, d):
    return c + a * numpy.exp(-x/d)
# %%
def plot_1D_SpaCE(file_name, file_path, do_plot = True, do_fit = False, 
                  do_save = True):
    data = tool_belt.get_raw_data( file_name, file_path)
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    CPG_pulse_dur = nv_sig['CPG_laser_dur']
    dir_1D = nv_sig['dir_1D']
    start_coords = nv_sig['coords']

    counts = data['readout_counts_avg']
    coords_voltages = data['coords_voltages']
    num_steps = data['num_steps']

    dir_1D = nv_sig['dir_1D']
    start_coords = nv_sig['coords']

    if dir_1D == 'x':
        coord_ind = 0
    elif dir_1D == 'y':
        coord_ind = 1
    voltages = [i[coord_ind] for i in coords_voltages]
    voltages = numpy.array(voltages)
    rad_dist = (voltages - start_coords[coord_ind])*35000
    opti_params = []


    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(rad_dist, counts, 'b.')
        if dir_1D == 'x':
            ax.set_xlabel('x (nm)')
        elif dir_1D == 'y':
            ax.set_xlabel('y (nm)')
        ax.set_ylabel('Average counts')
        ax.set_title('{} us pulse'.format(CPG_pulse_dur/10**3))
        
    if do_fit:
        init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
        try:
            opti_params, cov_arr = curve_fit(tool_belt.gaussian,
                  rad_dist,
                  counts,
                  p0=init_fit
                  )
            if do_plot:
                text = r'$C + A^2 e^{-(r - r_0)^2/(2*\sigma^2)}$'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                lin_radii = numpy.linspace(rad_dist[0],
                                rad_dist[-1], 100)
                ax.plot(lin_radii,
                       tool_belt.gaussian(lin_radii, *opti_params), 'r-')
                text = 'A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n ' \
                    '$\sigma$={:.3f} nm\nC={:.3f} counts'.format(*opti_params)
                ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
            print(opti_params)
        except Exception:
            text = 'Peak could not be fit'
            ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
    if do_plot and do_save:
        filePath = tool_belt.get_file_path(__file__, timestamp,
                                                nv_sig['name'])
        tool_belt.save_figure(fig, filePath + '-gaussian_fit')
            
                  

    return rad_dist, counts, opti_params
def gaussian_fit_1D_airy_rings(file_name, file_path, lobe_positions):
    rad_dist, counts, _ = plot_1D_SpaCE(file_name, file_path, do_plot = False)

    data = tool_belt.get_raw_data(file_name, file_path)
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    dir_1D = nv_sig['dir_1D']
    num_steps = data['num_steps']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    text = r'$C + A^2 e^{-(r - r_0)^2/(2*\sigma^2)}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_title('{} us pulse, power set to {}'.format(nv_sig['CPG_laser_dur']/10**3,
                                                       nv_sig['CPG_laser_power']))

    ax.plot(rad_dist, counts, 'b.')
    if dir_1D == 'x':
        ax.set_xlabel('x (nm)')
    elif dir_1D == 'y':
        ax.set_xlabel('y (nm)')
    ax.set_ylabel('Average counts')

    #get idea of where the center is
    zero_ind = int(num_steps/2)
    step_size_nm = rad_dist[1] - rad_dist[0]
    # Calculate the steps we'll consider around each ring position
    wings_ind = int( 400/step_size_nm)

    fit_params_list = []
    ring_positions = lobe_positions
    for ri in range(len(ring_positions)):
        fit_fail = False
        ring_r_nm = ring_positions[ri]
        if ring_r_nm > 0:
            ring_ind = int(zero_ind + ring_r_nm / step_size_nm)
        else:
            ring_ind = int(zero_ind - abs(ring_r_nm / step_size_nm))
        ring_range = [ring_ind-wings_ind,
                            ring_ind+wings_ind]
        # ring_ind_high = int(zero_ind + ring_r_nm / step_size_nm)
        # ring_range_high = [ring_ind_high-wings_ind,
        #                     ring_ind_high+wings_ind]
        # ring_ind_low = int(zero_ind - ring_r_nm / step_size_nm)
        # ring_range_low = [ring_ind_low-wings_ind,
        #                     ring_ind_low+wings_ind]
        init_fit = [2, ring_r_nm, 15, 7]
        try:
            opti_params, cov_arr = curve_fit(tool_belt.gaussian,
                  rad_dist[ring_range[0]: ring_range[1]],
                  counts[ring_range[0]: ring_range[1]],
                  p0=init_fit,
                    bounds = ([-numpy.infty, -(abs(ring_r_nm) + 10), -80, 0],
                              [numpy.infty, abs(ring_r_nm) + 10, 80, 11])
                  )

            lin_radii = numpy.linspace(rad_dist[ring_range[0]],
                            rad_dist[ring_range[1]], 100)
            ax.plot(lin_radii,
                   tool_belt.gaussian(lin_radii, *opti_params), 'r-')
            text = 'A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n ' \
                '$\sigma$={:.3f} nm\nC={:.3f} counts'.format(*opti_params)
            ax.text(0.3*(ri), 0.1, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            print(opti_params)
        except Exception:
            fit_fail = True
            text = 'Peak could not be\nfit for +{} nm lobe'.format(ring_r_nm)
            ax.text(0.3*ri, 0.1, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

        if not fit_fail:
            fit_params_list.append(opti_params)
        else:
            fit_params_list.append(None)

        # init_fit = [4, -ring_r_nm, 15, 3]
        # try:
        #     opti_params, cov_arr = curve_fit(tool_belt.gaussian,
        #           rad_dist[ring_range_low[0]: ring_range_low[1]],
        #           counts[ring_range_low[0]: ring_range_low[1]],
        #           p0=init_fit,
        #           bounds = ([-numpy.infty, -numpy.infty, -numpy.infty, 0],
        #                     [numpy.infty, 0, 100, 9]))

        #     lin_radii = numpy.linspace(rad_dist[ring_range_low[0]],
        #                         rad_dist[ring_range_low[1]], 100)
        #     ax.plot(lin_radii,
        #            tool_belt.gaussian(lin_radii, *opti_params), 'r-')
        #     text = 'A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n ' \
        #         '$\sigma$={:.3f} nm\nC={:.3f} counts'.format(*opti_params)
        #     ax.text(0.5-0.3*(1+ri), 0.1, text, transform=ax.transAxes, fontsize=12,
        #             verticalalignment='top', bbox=props)
        #     print(opti_params)
        # except Exception:
        #     text = 'Peak could not be fit for -{} nm lobe'.format(ring_r_nm)
        #     ax.text(0.5-0.3*(1+ri), 0.1, text, transform=ax.transAxes, fontsize=12,
        #             verticalalignment='top', bbox=props)


    filePath = tool_belt.get_file_path(__file__, timestamp,
                                            nv_sig['name'])
    tool_belt.save_figure(fig, filePath + '-gaussian_fit')

    return fit_params_list

# %%
def build_voltages_from_list(start_coords_drift, coords_list_drift):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]

    num_samples = len(coords_list_drift)

    # we want this list to have the pattern [[readout], [target], [readout], [readout],
    #                                                   [target], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]

    # now create a list of all the coords we want to feed to the galvo
    for i in range(num_samples):
        x_points.append(coords_list_drift[i][0])
        x_points.append(start_x_value)
        x_points.append(start_x_value)

        y_points.append(coords_list_drift[i][1])
        y_points.append(start_y_value)
        y_points.append(start_y_value)

    return x_points, y_points

def build_xy_voltages_w_optimize(start_coords, CPG_coords,
                              num_opti_steps, opti_scan_range):
    # the start value is the nv's positions
    start_x_value = start_coords[0]
    start_y_value = start_coords[1]

    ################# SPaCE measurement #################
    x_points = [start_x_value, CPG_coords[0], start_x_value]
    y_points = [start_y_value, CPG_coords[1], start_y_value]

    ################# optimization #################
    half_scan_range = opti_scan_range / 2

    x_low = start_x_value - half_scan_range
    x_high = start_x_value + half_scan_range

    y_low = start_y_value - half_scan_range
    y_high = start_y_value + half_scan_range

    # build x and y values for scan in x
    x_voltages = numpy.linspace(x_low, x_high, num_opti_steps).tolist()
    y_voltages = numpy.full(num_opti_steps, start_y_value).tolist()

    x_points = x_points + x_voltages
    y_points = y_points + y_voltages

    # build x and y values for scan in y
    x_voltages = numpy.full(num_opti_steps, start_x_value).tolist()
    y_voltages = numpy.linspace(y_low, y_high, num_opti_steps).tolist()

    x_points = x_points + x_voltages
    y_points = y_points + y_voltages

    return x_points, y_points

def build_voltages_image(start_coords, img_range, num_steps):
    x_center = start_coords[0]
    y_center = start_coords[1]

    x_num_steps = num_steps
    y_num_steps = num_steps

    # Force the scan to have square pixels by only applying num_steps
    # to the shorter axis
    half_x_range = img_range / 2
    half_y_range = img_range / 2

    x_low = x_center - half_x_range
    x_high = x_center + half_x_range
    y_low = y_center - half_y_range
    y_high = y_center + half_y_range

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions
    # are linear in these voltages. For a small range, however, we don't
    # really care.
    x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
    y_voltages_1d = numpy.linspace(y_low, y_high, num_steps)

    # Winding cartesian product
    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    x_inter = numpy.concatenate((x_voltages_1d,
                                 numpy.flipud(x_voltages_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if y_num_steps % 2 == 0:  # Even x size
        target_x_values = numpy.tile(x_inter, int(y_num_steps/2))
    else:  # Odd x size
        target_x_values = numpy.tile(x_inter, int(numpy.floor(y_num_steps/2)))
        target_x_values = numpy.concatenate((target_x_values, x_voltages_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    target_y_values = numpy.repeat(y_voltages_1d, x_num_steps)

    return target_x_values, target_y_values, x_voltages_1d, y_voltages_1d
# %%
def populate_img_array(valsToAdd, imgArray, run_num):
    """
    We scan the sample in a winding pattern. This function takes a chunk
    of the 1D list returned by this process and places each value appropriately
    in the 2D image array. This allows for real time imaging of the sample's
    fluorescence.

    Note that this function could probably be much faster. At least in this
    context, we don't care if it's fast. The implementation below was
    written for simplicity.

    Params:
        valsToAdd: numpy.ndarray
            The increment of raw data to add to the image array
        imgArray: numpy.ndarray
            The xDim x yDim array of fluorescence counts
        writePos: tuple(int)
            The last x, y write position on the image array. [] will default
            to the bottom right corner. Third index is the run number
    """
    yDim = imgArray.shape[0]
    xDim = imgArray.shape[1]

    # Start with the write position at the start
    writePos = [xDim, yDim - 1, run_num]

    xPos = writePos[0]
    yPos = writePos[1]

    # Figure out what direction we're heading
    headingLeft = ((yDim - 1 - yPos) % 2 == 0)

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if (xPos == 0):
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos, run_num] = val
        else:
            # Determine if we're at the right x edge
            if (xPos == xDim - 1):
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos + 1
                imgArray[yPos, xPos, run_num] = val
    return
# %%
def data_collection_optimize(nv_sig, coords_list,run_num,  opti_plot  = False):
    with labrad.connect() as cxn:
        ret_vals = data_collection_optimize_with_cxn(cxn, nv_sig, coords_list,
                                                     run_num, opti_plot)

    readout_counts_array, opti_coords_list = ret_vals

    return readout_counts_array,  opti_coords_list

def data_collection_optimize_with_cxn(cxn, nv_sig, coords_list, run_num,
                                      opti_plot  = False):
    '''
    Runs a measurement where an initial pulse is pulsed on the start coords,
    then a pulse is set on the first point in the coords list, then the
    counts are recorded on the start coords. The routine steps through
    the coords list

    Here, we run each point individually, and we optimize before each point to
    ensure we're centered on the NV. The optimize function is built into the
    sequence.

    Parameters
    ----------
    cxn :
        labrad connection. See other our other python functions.
    nv_sig : dict
        dictionary containing onformation about the pulse lengths, pusle powers,
        expected count rate, nd filter, color filter, etc
    coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.

    Returns
    -------
    readout_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate
        measured on the start coord.
        The first index refers to the coordinate, the secon index refers to the
        run.
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    '''
    tool_belt.reset_cfm(cxn)
    gaussian_fit = optimize.fit_gaussian
    xyz_server = tool_belt.get_xyz_server(cxn)

    # Define paramters
    apd_indices = [0]
    drift_list = []
    num_opti_steps = 21#31
    # There will be three samples from the SPaCE measurement, folllowed by
    # num_opti_steps sampels for each three optimize axes.
    total_num_samples = 3 + 2 * num_opti_steps

    xy_opti_scan_range = 2/3 * (tool_belt.get_registry_entry_no_cxn('xy_optimize_range',
                           ['Config','Positioning']))

    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    pulse_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['charge_readout_laser']])
    x_move_delay = tool_belt.get_registry_entry_no_cxn('xy_large_response_delay',
                           ['Config','Positioning'])
    y_move_delay = tool_belt.get_registry_entry_no_cxn('xy_large_response_delay',
                           ['Config','Positioning'])

    pulse_time = nv_sig['CPG_laser_dur']
    initialization_time = nv_sig['initialize_dur']
    charge_readout_time = nv_sig['charge_readout_dur']
    charge_readout_laser_power = nv_sig['charge_readout_laser_power']
    imaging_readout_dur = nv_sig['imaging_readout_dur']

    num_samples = len(coords_list)
    start_coords = nv_sig['coords']

    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')

    # Readout array will be a list in this case. This will be a list with
    # dimensions [num_samples].
    readout_counts_list = []

    # optimize before the start of the measurement
    optimize.main_with_cxn(cxn, nv_sig, apd_indices)

    # define the sequence paramters
    file_name = 'SPaCE_w_optimize_xy.py'
    # file_name = 'SPaCE.py'
    seq_args = [ initialization_time, pulse_time, charge_readout_time,
                imaging_readout_dur, x_move_delay, y_move_delay,
                charge_readout_laser_power, num_opti_steps, apd_indices[0],
                init_color, pulse_color, readout_color]
    # seq_args = [ initialization_time, pulse_time, charge_readout_time,
    #     charge_readout_laser_power,
    #     apd_indices[0],
    #     init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    # return

    # print the expected run time
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*num_samples + 1)
    print('{} ms pulse time'.format(pulse_time/10**6))
    print('Expected run time for set of points: {:.1f} m'.format(period_s_total/60))
    # load the sequence
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)


    for i in range(num_samples):
        print("Run {}, point {}/{}".format(run_num, i, num_samples))
        # Get the current drift
        drift = numpy.array(tool_belt.get_drift())

        # get the readout coords with drift
        start_coords_drift = start_coords + drift
        coords_list_drift = numpy.array(coords_list) + [drift[0], drift[1]]

        # step thru the coordinates to test as the cpg pulse
        CPG_coord = [coords_list_drift[i][0], coords_list_drift[i][1],
                     start_coords_drift[2]]
        
        # Build the x,y, and z coordinate lists, which change with each CLK pulse
        x_voltages, y_voltages = build_xy_voltages_w_optimize(start_coords_drift,
                      CPG_coord, num_opti_steps, xy_opti_scan_range)
        # x_voltages, y_voltages = build_voltages_from_list(start_coords_drift, [CPG_coord])

        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_drift)

        # Load the galvo and objective piezo server
        xyz_server.load_arb_xy_scan(x_voltages, y_voltages, int(period))

        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        cxn.pulse_streamer.stream_start()


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        
        # # We'll be lookign for three samples each repetition with how I have
        # # the sequence set up
        # total_num_samples = 3
    
        # total_samples_list = []
        # num_read_so_far = 0
        # tool_belt.init_safe_stop()
    
        # while num_read_so_far < total_num_samples:
    
        #     if tool_belt.safe_stop():
        #         break
    
        #     # Read the samples and update the image
        #     new_samples = cxn.apd_tagger.read_counter_simple()
        #     num_new_samples = len(new_samples)
    
        #     if num_new_samples > 0:
        #         for el in new_samples:
        #             total_samples_list.append(int(el))
        #         num_read_so_far += num_new_samples
    
        # # The last of the triplet of readout windows is the counts we are interested in
        # readout_counts = int(total_samples_list[2])
        # readout_counts_list.append(int(readout_counts))
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    
        total_samples_list = []
        num_read_so_far = 0
        tool_belt.init_safe_stop()



        while num_read_so_far < total_num_samples:

            if tool_belt.safe_stop():
                break

            # Read the samples and update the image
            new_samples = cxn.apd_tagger.read_counter_simple()
            num_new_samples = len(new_samples)

            if num_new_samples > 0:
                for el in new_samples:
                    total_samples_list.append(int(el))
                num_read_so_far += num_new_samples

        # The third value is the charge readout.
        charge_readout_count = total_samples_list[2]
        readout_counts_list.append(charge_readout_count)
        x_opti_counts = total_samples_list[3:num_opti_steps+3]
        y_opti_counts = total_samples_list[num_opti_steps+3:2*num_opti_steps+3]

        #fit to the x, y, and z lists, find drift, update drift. etc
        x_scan_voltages = x_voltages[3:num_opti_steps+3]
        y_scan_voltages = y_voltages[num_opti_steps+3:2*num_opti_steps+3]

        # IF you want to see the optimize counts every time, set opti_plot to True
        if opti_plot:
            fig = optimize.create_figure()
            optimize.update_figure(fig, 0, x_scan_voltages, x_opti_counts)
            optimize.update_figure(fig, 1, y_scan_voltages, y_opti_counts)

            x_opti_coord = gaussian_fit(nv_sig, x_scan_voltages, x_opti_counts, 0, fig)
            y_opti_coord = gaussian_fit(nv_sig, y_scan_voltages, y_opti_counts, 1, fig)
        else:
            x_opti_coord = gaussian_fit(nv_sig, x_scan_voltages, x_opti_counts, 0)
            y_opti_coord = gaussian_fit(nv_sig, y_scan_voltages, y_opti_counts, 1)


        opti_coords = [x_opti_coord, y_opti_coord, start_coords_drift[2]]
        #If optimize failed, jsut set that coordinate to the start coordinate
        for n, el in enumerate(opti_coords):
            if el == None:
                opti_coords[n] = start_coords[n]

        # print(opti_coords)
        # print(start_coords)
        drift = (numpy.array(opti_coords) - numpy.array(start_coords)).tolist()
        tool_belt.set_drift(drift)
        # print(drift)

        drift_list.append(drift)



    return readout_counts_list, drift_list
       # %%
def main_data_collection(nv_sig, coords_list):
    with labrad.connect() as cxn:
        ret_vals = main_data_collection_with_cxn(cxn, nv_sig, coords_list)

    readout_counts_array, opti_coords_list = ret_vals

    return readout_counts_array,  opti_coords_list

def main_data_collection_with_cxn(cxn, nv_sig, coords_list):
    '''
    Runs a measurement where an initial pulse is pulsed on the start coords,
    then a pulse is set on the first point in the coords list, then the
    counts are recorded on the start coords. The routine steps through
    the coords list

    Parameters
    ----------
    cxn :
        labrad connection. See other our other python functions.
    nv_sig : dict
        dictionary containing onformation about the pulse lengths, pusle powers,
        expected count rate, nd filter, color filter, etc
    coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.

    Returns
    -------
    readout_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate
        measured on the start coord.
        The first index refers to the coordinate, the secon index refers to the
        run.
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    '''
    tool_belt.reset_cfm(cxn)

    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    pulse_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['charge_readout_laser']])
    pulse_time = nv_sig['CPG_laser_dur']
    initialization_time = nv_sig['initialize_dur']
    readout_pulse_time = nv_sig['charge_readout_dur']
    charge_readout_laser_power = nv_sig['charge_readout_laser_power']

    # Define paramters
    apd_indices = [0]
    # green_2mw_power = 0.655

    num_samples = len(coords_list)
    start_coords = nv_sig['coords']

    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')



    # Readout array will be a list in this case. This will be a list with
    # dimensions [num_samples].
    readout_counts_list = []


    # define the sequence paramters
    file_name = 'SPaCE.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time,
        charge_readout_laser_power,
        apd_indices[0],
        init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    # print(seq_args)
    # return

    # print the expected run time
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*num_samples + 1)
    print('{} ms pulse time'.format(pulse_time/10**6))
    print('Expected total run time: {:.1f} m'.format(period_s_total/60))

    # Optimize at the start of the routine
    # Set up a timed optimize--- every 2 min.
    time_now = time.time()
    global time_start
    
    if time_now - time_start > 2 * 60:
        optimize.main_with_cxn(cxn, nv_sig, apd_indices)

        time_start = time_now

    drift = numpy.array(tool_belt.get_drift())
    # get the readout coords with drift
    start_coords_drift = start_coords + drift
    coords_list_drift = numpy.array(coords_list) + [drift[0], drift[1]]

        # start on the readout NV
    tool_belt.set_xyz(cxn, start_coords_drift)

    # load the sequence
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Build the list to step through the coords on readout NV and targets
    x_voltages, y_voltages = build_voltages_from_list(start_coords_drift, coords_list_drift)

    # Load the galvo
    xy_server = tool_belt.get_xy_server(cxn)
    xy_server.load_arb_xy_scan(x_voltages, y_voltages, int(period))

    #  Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)

    cxn.pulse_streamer.stream_start(num_samples)

    # We'll be lookign for three samples each repetition with how I have
    # the sequence set up
    total_num_samples = 3*num_samples

    total_samples_list = []
    num_read_so_far = 0
    tool_belt.init_safe_stop()

    while num_read_so_far < total_num_samples:

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
        num_new_samples = len(new_samples)

        if num_new_samples > 0:
            for el in new_samples:
                total_samples_list.append(int(el))
            num_read_so_far += num_new_samples

    # The last of the triplet of readout windows is the counts we are interested in
    readout_counts = total_samples_list[2::3]
    readout_counts_list = [int(el) for el in readout_counts]

    return readout_counts_list, drift

# %%
def main(nv_sig, img_range, num_steps, num_runs, measurement_type):
    '''
    A measurements to initialize on a single point, then pulse a laser off that
    point, and then read out the charge state on the single point.

    Parameters
    ----------
    nv_sig : dict
        dictionary specific to the nv, which contains parameters liek the
        scc readout length, the yellow readout power, the expected count rate...\
    img_range: float
        the range that the 2D area will cover, in both x and y. NOTE: for 1D
        measurement, this is divided in half and used as the farthest point
        in the +x direction
    num_steps: int
        number of steps in 1 direction. For 2D image, this is squared for the
        total number of points
    num_runs: int
        the number of repetitions of the same measurement, which will be averaged over
    measurement_type: '1D' or '2D'
        either run a 2D oving target or 1D moving target measurement.
    '''
    # Record start time of the measurement
    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    pulse_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    pulse_time = nv_sig['CPG_laser_dur']

    start_coords = nv_sig['coords']

    if measurement_type == '1D':
        dir_1D = nv_sig['dir_1D']
        if type(img_range) == float:
            dr = img_range / 2
            if dir_1D == 'x':
                low_coords = numpy.array(start_coords) - [dr, 0, 0]
                high_coords = numpy.array(start_coords) + [dr, 0, 0]
            elif dir_1D == 'y':
                low_coords = numpy.array(start_coords) - [0, dr, 0]
                high_coords = numpy.array(start_coords) + [0, dr, 0]
            # end_coords = numpy.array(start_coords) + [dx, 0, 0]
            # calculate the x and y values for linearly spaced points between start and end
            x_voltages = numpy.linspace(low_coords[0],
                                        high_coords[0], num_steps)
            y_voltages = numpy.linspace(low_coords[1],
                                        high_coords[1], num_steps)
        elif type(img_range) == list:
            # Make sure the lower value is first.
            img_range.sort()
            if dir_1D == 'x':
                low_coords = numpy.array(start_coords) + [img_range[0], 0, 0]
                high_coords = numpy.array(start_coords) + [img_range[1], 0, 0]
            elif dir_1D == 'y':
                low_coords = numpy.array(start_coords) + [0, img_range[0], 0]
                high_coords = numpy.array(start_coords) + [0, img_range[1], 0]
            x_voltages = numpy.linspace(low_coords[0],
                                        high_coords[0], num_steps)
            y_voltages = numpy.linspace(low_coords[1],
                                        high_coords[1], num_steps)
        # Zip the two list together
        coords_voltages = list(zip(x_voltages, y_voltages))

        # calculate the radial distances from the readout NV to the target points
        if dir_1D == 'x':
            rad_dist = (x_voltages - start_coords[0])
        elif dir_1D == 'y':
            rad_dist = (y_voltages- start_coords[1])


    elif measurement_type == '2D':
        # calculate the list of x and y voltages we'll need to step through
        ret_vals= build_voltages_image(start_coords, img_range, num_steps)
        x_voltages, y_voltages, x_voltages_1d, y_voltages_1d  = ret_vals

        # Combine the x and y voltages together into pairs
        coords_voltages = list(zip(x_voltages, y_voltages))
        num_samples = len(coords_voltages)

        # prep for the figure
        x_low = x_voltages_1d[0]
        x_high = x_voltages_1d[num_steps-1]
        y_low = y_voltages_1d[0]
        y_high = y_voltages_1d[num_steps-1]

        pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])

        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]


        # Create some empty data lists

        readout_image_array = numpy.empty([num_steps, num_steps])
        readout_image_array[:] = numpy.nan

        # Create the figure
        title = 'SPaCE - {} nm init pulse \n{} nm {} ms CPG pulse'.format(init_color, pulse_color, pulse_time/10**6)
        fig_2D = tool_belt.create_image_figure(readout_image_array,
                                               numpy.array(img_extent)*35,
                                                title = title, um_scaled = True)

    drift_list_master = []
    num_samples = len(coords_voltages)
    readout_counts_array = numpy.empty([num_samples, num_runs])

    for n in range(num_runs):
        print('Run {}'.format(n))
        # shuffle the voltages that we're stepping thru
        ind_list = list(range(num_samples))
        shuffle(ind_list)

        # shuffle the voltages to run
        coords_voltages_shuffle = []
        for i in ind_list:
            coords_voltages_shuffle.append(coords_voltages[i])
        coords_voltages_shuffle_list = [list(el) for el in coords_voltages_shuffle]

        #========================== Run the data collection====================
        # ret_vals = data_collection_optimize(nv_sig, coords_voltages_shuffle_list, n, opti_plot=False)
        ret_vals = main_data_collection(nv_sig, coords_voltages_shuffle_list)

        readout_counts_list_shfl, drift = ret_vals
        drift_list_master.append(drift)
        readout_counts_list_shfl = numpy.array(readout_counts_list_shfl)

        # unshuffle the raw data
        list_ind = 0
        for f in ind_list:
            readout_counts_array[f][n] = readout_counts_list_shfl[list_ind]
            list_ind += 1

        # Take the average and ste. Need to rotate the matrix, to then only
        # average the runs that have been completed
        readout_counts_array_rot = numpy.rot90(readout_counts_array)
        readout_counts_avg = numpy.average(readout_counts_array_rot[-(n+1):], axis = 0)
        readout_counts_ste = stats.sem(readout_counts_array_rot[-(n+1):], axis = 0)
        #Save incrementally
        raw_data = {'timestamp': start_timestamp,
                'img_range': img_range,
                'img_range-units': 'V',
                'num_steps': num_steps,
                'num_runs':num_runs,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'coords_voltages': coords_voltages,
                'coords_voltages-units': '[V, V]',
                 'ind_list': ind_list,
                 'drift_list_master': numpy.array(drift_list_master).tolist(),
                'readout_counts_array': readout_counts_array.tolist(),
                'readout_counts_array-units': 'counts',
                'readout_counts_avg': readout_counts_avg.tolist(),
                'readout_counts_avg-units': 'counts',
                'readout_counts_ste': readout_counts_ste.tolist(),
                'readout_counts_ste-units': 'counts',
                }
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')

        if measurement_type == '2D':
            # create the img arrays
            writePos = []
            readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)

            tool_belt.update_image_figure(fig_2D, readout_image_array)

            raw_data['x_voltages_1d'] = x_voltages_1d.tolist()
            raw_data['y_voltages_1d'] = y_voltages_1d.tolist()
            raw_data['img_extent'] = img_extent
            raw_data['img_extent-units'] = 'V'
            raw_data['readout_image_array'] = readout_image_array.tolist()
            raw_data['readout_counts_array-units'] = 'counts'


            tool_belt.save_figure(fig_2D, file_path)

        tool_belt.save_raw_data(raw_data, file_path)

    # Save at the end of the whole measurement
    # measure laser powers:
    # green_optical_power_pd, green_optical_power_mW, \
    #         red_optical_power_pd, red_optical_power_mW, \
    #         yellow_optical_power_pd, yellow_optical_power_mW = \
    #         tool_belt.measure_g_r_y_power(
    #                           nv_sig['am_589_power'], nv_sig['nd_filter'])

    endFunctionTime = time.time()

    # Save
    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
            'img_range': img_range,
            'img_range-units': 'V',
            'num_steps': num_steps,
            'num_runs':num_runs,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            # 'green_optical_power_pd': green_optical_power_pd,
            # 'green_optical_power_pd-units': 'V',
            # 'green_optical_power_mW': green_optical_power_mW,
            # 'green_optical_power_mW-units': 'mW',
            # 'red_optical_power_pd': red_optical_power_pd,
            # 'red_optical_power_pd-units': 'V',
            # 'red_optical_power_mW': red_optical_power_mW,
            # 'red_optical_power_mW-units': 'mW',
            # 'yellow_optical_power_pd': yellow_optical_power_pd,
            # 'yellow_optical_power_pd-units': 'V',
            # 'yellow_optical_power_mW': yellow_optical_power_mW,
            # 'yellow_optical_power_mW-units': 'mW',
            'coords_voltages': coords_voltages,
            'coords_voltages-units': '[V, V]',
             'ind_list': ind_list,
            'drift_list_master': numpy.array(drift_list_master).tolist(),
            'readout_counts_array': readout_counts_array.tolist(),
            'readout_counts_array-units': 'counts',
            'readout_counts_avg': readout_counts_avg.tolist(),
            'readout_counts_avg-units': 'counts',
            'readout_counts_ste': readout_counts_ste.tolist(),
            'readout_counts_ste-units': 'counts',
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])

    if measurement_type == '1D':
        fig_1D, ax_1D = plt.subplots(1, 1, figsize=(10, 10))
        ax_1D.plot(rad_dist*35000,readout_counts_avg, label = nv_sig['name'])
        if dir_1D == 'x':
            ax_1D.set_xlabel('x (nm)')
        elif dir_1D == 'y':
            ax_1D.set_xlabel('y (nm)')
        ax_1D.set_ylabel('Average counts')
        ax_1D.set_title('SPaCE - {} nm init pulse \n{} nm {} ms CPG pulse'.\
                                        format(init_color, pulse_color, pulse_time/10**6,))
        ax_1D.legend()
        tool_belt.save_figure(fig_1D, file_path)

    if measurement_type == '2D':
            raw_data['x_voltages_1d'] = x_voltages_1d.tolist()
            raw_data['y_voltages_1d'] = y_voltages_1d.tolist()
            raw_data['img_extent'] = img_extent
            raw_data['img_extent-units'] = 'V'
            raw_data['readout_image_array'] = readout_image_array.tolist()
            raw_data['readout_counts_array-units'] = 'counts'


            tool_belt.save_figure(fig_2D, file_path)

    tool_belt.save_raw_data(raw_data, file_path)

    return


# %% Run the files

if __name__ == '__main__':

    path_july = 'pc_rabi/branch_master/SPaCE/2021_07'
    path = 'pc_rabi/branch_master/SPaCE/2021_08'
    # sub_folder = '2021_08_02 22 mW vary pulse dur/y'
    sub_folder = '2021_08_02 1000 us vary pulse power/y'
    # file_name = '2021_07_26-22_23_54-johnson-nv1_2021_07_21'
    # file_name = '2021_07_27-09_03_48-johnson-nv1_2021_07_21'
    # file_name = '2021_07_26-23_35_05-johnson-nv1_2021_07_21'

    file_name_no_opt = '2021_07_27-15_10_17-johnson-nv1_2021_07_27'
    file_name_no_opt_2 = '2021_07_28-15_21_25-johnson-nv1_2021_07_27'
    file_name_no_opt_dim = '2021_08_03-15_13_57-johnson-nv1_2021_07_27'
    file_name_opt = '2021_07_27-19_03_17-johnson-nv1_2021_07_27'
    file_name_opt_xy = '2021_07_28-14_00_23-johnson-nv1_2021_07_27'
    file_name_opt_mod = '2021_08_03-12_54_18-johnson-nv1_2021_07_27'
    file_name_opt_dim = '2021_08_03-14_48_39-johnson-nv1_2021_07_27'
    
    file_name_no_opt_400_ms = '2021_08_03-16_21_50-johnson-nv1_2021_07_27'
    file_name_opt_xy_2 = '2021_08_03-18_22_18-johnson-nv1_2021_07_27'
    
    file = '2021_07_31-18_02_29-johnson-nv1_2021_07_27'

    #================ 7/27/2021 x and y scans @ 22 mW ================#
    # file_list = ['2021_07_27-04_19_32-johnson-nv1_2021_07_21', # x
    #               '2021_07_27-01_57_21-johnson-nv1_2021_07_21',
    #               '2021_07_26-23_35_05-johnson-nv1_2021_07_21',
    #               '2021_07_26-21_12_36-johnson-nv1_2021_07_21']


    # file_list = ['2021_07_27-05_30_36-johnson-nv1_2021_07_21', # y
    #               '2021_07_27-03_08_29-johnson-nv1_2021_07_21',
    #               '2021_07_27-00_46_16-johnson-nv1_2021_07_21',
    #               '2021_07_26-22_23_54-johnson-nv1_2021_07_21']

    #================ 7/28/2021 x and y scans @ 33 mW ================#
    # x
    # file_list = ['2021_07_28-00_46_22-johnson-nv1_2021_07_27',
                  # '2021_07_28-03_08_32-johnson-nv1_2021_07_27',
                  # '2021_07_28-05_30_52-johnson-nv1_2021_07_27',
                  # '2021_07_28-07_54_25-johnson-nv1_2021_07_27',
                  # '2021_07_27-21_13_36-johnson-nv1_2021_07_27'
                  # ]

    # y
    # file_list = ['2021_07_28-01_57_24-johnson-nv1_2021_07_27',
    #              '2021_07_28-04_19_38-johnson-nv1_2021_07_27',
    #              '2021_07_28-06_42_07-johnson-nv1_2021_07_27',
    #              '2021_07_28-09_06_42-johnson-nv1_2021_07_27',
    #              '2021_07_27-22_39_56-johnson-nv1_2021_07_27']


    #================ 8/2/2021 x and y scans @ 22 mW ================#
    # x
    # file_list = [
    #             '2021_07_30-18_15_21-johnson-nv1_2021_07_27',
    #                '2021_07_30-20_39_48-johnson-nv1_2021_07_27',
    #                '2021_07_31-10_49_04-johnson-nv1_2021_07_27',
    #               '2021_07_31-23_41_32-johnson-nv1_2021_07_27',
    #               '2021_07_31-13_13_29-johnson-nv1_2021_07_27',
    #               '2021_07_31-15_37_56-johnson-nv1_2021_07_27',
    #               '2021_07_31-18_02_29-johnson-nv1_2021_07_27',
    #               '2021_08_01-02_06_04-johnson-nv1_2021_07_27',
    #               '2021_08_01-04_30_39-johnson-nv1_2021_07_27',
    #               '2021_08_01-06_55_19-johnson-nv1_2021_07_27',
    #               '2021_08_01-09_20_02-johnson-nv1_2021_07_27',
    #               '2021_08_01-11_44_49-johnson-nv1_2021_07_27'
    #                ]

    # y
    # file_list = [
    #             '2021_07_30-19_27_35-johnson-nv1_2021_07_27',
    #                '2021_07_30-21_52_01-johnson-nv1_2021_07_27',
    #                '2021_07_31-12_01_16-johnson-nv1_2021_07_27',
    #                '2021_08_01-00_53_48-johnson-nv1_2021_07_27',
    #                '2021_07_31-14_25_41-johnson-nv1_2021_07_27',
    #                '2021_07_31-16_50_12-johnson-nv1_2021_07_27',
    #                '2021_07_31-19_14_50-johnson-nv1_2021_07_27',
    #                '2021_08_01-03_18_22-johnson-nv1_2021_07_27',
    #                '2021_08_01-05_42_58-johnson-nv1_2021_07_27',
    #                '2021_08_01-08_07_40-johnson-nv1_2021_07_27',
    #                '2021_08_01-10_32_24-johnson-nv1_2021_07_27',
    #                '2021_08_01-12_57_13-johnson-nv1_2021_07_27',
    #             ]
    
    #================ 8/2/2021 x scans @ 800 us ================#
    
    # x
    # file_list = [
    #     '2021_08_02-14_21_03-johnson-nv1_2021_07_27',
    #     '2021_08_01-06_55_19-johnson-nv1_2021_07_27',
    #     '2021_08_01-23_52_30-johnson-nv1_2021_07_27',
    #     '2021_08_01-14_39_32-johnson-nv1_2021_07_27'
    #     ]
    
    #================ 8/2/2021 y scans @ 1000 us ================#
    
    # y
    # file_list = [
    #     '2021_08_01-12_57_13-johnson-nv1_2021_07_27',
    #     '2021_08_02-03_29_37-johnson-nv1_2021_07_27',
    #     '2021_08_01-18_16_42-johnson-nv1_2021_07_27'
    #     ]
    #================ 8/5/2021 x scans @ 22 mW Feature 1A ================#
    # x
    file_list = [
        '2021_08_04-20_38_12-johnson-nv2_2021_08_04',
        '2021_08_04-21_29_32-johnson-nv2_2021_08_04',
        '2021_08_04-21_55_15-johnson-nv2_2021_08_04',
        '2021_08_04-22_20_56-johnson-nv2_2021_08_04',
        '2021_08_04-22_46_41-johnson-nv2_2021_08_04',
        '2021_08_04-23_12_20-johnson-nv2_2021_08_04',
        '2021_08_04-23_38_01-johnson-nv2_2021_08_04',
        '2021_08_05-00_03_43-johnson-nv2_2021_08_04',
        '2021_08_05-00_29_24-johnson-nv2_2021_08_04',
        '2021_08_05-17_59_06-johnson-nv2_2021_08_04',
        '2021_08_05-00_55_06-johnson-nv2_2021_08_04',
        '2021_08_05-01_20_49-johnson-nv2_2021_08_04',
        '2021_08_05-01_46_33-johnson-nv2_2021_08_04',
        '2021_08_05-02_12_17-johnson-nv2_2021_08_04',
        '2021_08_05-02_38_00-johnson-nv2_2021_08_04',
        '2021_08_05-03_03_43-johnson-nv2_2021_08_04',
        '2021_08_05-03_29_28-johnson-nv2_2021_08_04',
        '2021_08_05-03_55_11-johnson-nv2_2021_08_04',
        '2021_08_05-09_01_01-johnson-nv2_2021_08_04',
        '2021_08_05-09_26_52-johnson-nv2_2021_08_04',
        '2021_08_05-09_52_38-johnson-nv2_2021_08_04',
        '2021_08_05-10_45_26-johnson-nv2_2021_08_04',
        '2021_08_05-11_43_11-johnson-nv2_2021_08_04',
        '2021_08_05-12_12_58-johnson-nv2_2021_08_04',
        '2021_08_05-15_54_52-johnson-nv2_2021_08_04'
        ]

    ########### Fit Gaussian to 1D files ###########
    widths_master_list = []
    center_master_list = []
    height_master_list = []
    for file_name in file_list:

        lobe_positions = [750] # 400, 800, 1200, 1600
        ret_vals = plot_1D_SpaCE(file_name, path, do_plot = True, do_fit = True)
        widths_master_list.append(ret_vals[2][2])
        center_master_list.append(ret_vals[2][1])
        height_master_list.append(ret_vals[2][0]**2)

    # x_vals = [ 22, 25, 33]
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(x_vals, widths_master_list, 'go')
    # ax.set_xlabel('Pulse power (mW)')
    # ax.set_ylabel('Gaussian sigma (nm)')
    # ax.set_title('8/2/2021 1000 us, y axis, -770 nm lobe')
    
    x_vals = [90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 
              650, 700, 750, 800,850 , 900, 1000, 1100, 1200, 1300, 
              1350, 1400, 1450]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(x_vals, height_master_list, 'bo')
    ax.set_xlabel('Pulse duration (us)')
    ax.set_ylabel('Gaussian amplitude (counts)')
    ax.set_title('8/5/2021 22 mW, x axis, 1A lobe')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(x_vals, widths_master_list, 'bo')
    ax.set_xlabel('Pulse duration (us)')
    ax.set_ylabel('Gaussian sigma (nm)')
    ax.set_title('8/5/2021 22 mW, x axis, 1A lobe')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(x_vals, center_master_list, 'bo')
    ax.set_xlabel('Pulse duration (us)')
    ax.set_ylabel('Lobe position (nm)')
    ax.set_title('8/5/2021 22 mW, x axis, 1A lobe')


    ############# Plot 1D comparisons ##############
    # rad_dist, counts_no_opt = plot_1D_SpaCE(file_name_no_opt_400_ms, path, do_plot = False)
    # rad_dist, counts_opt = plot_1D_SpaCE(file_name_opt_xy_2, path, do_plot = False)
    
    
    # rad_dist, counts = plot_1D_SpaCE(file, path_july, do_plot = False)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # # ax.plot(rad_dist, counts, 'b-', label = 'Optimize every run')
    # ax.plot(rad_dist, counts_no_opt, 'b-', label = 'Optimize every run, add 400 ms of green pusle every point')
    # ax.plot(rad_dist, counts_opt, 'r-', label = 'Optimize every point in x and y')
    # ax.set_xlabel('y (nm)')
    # ax.set_ylabel('Average counts')
    # ax.legend()




    # specific for 2D scans
    # try:
    #     img_array = data['readout_image_array']
    #     x_voltages = data['x_voltages_1d']
    #     y_voltages = data['y_voltages_1d']
    #     x_low = x_voltages[0]
    #     x_high = x_voltages[-1]
    #     y_low = y_voltages[0]
    #     y_high = y_voltages[-1]
    #     pixel_size = x_voltages[1] - x_voltages[0]
    #     half_pixel_size = pixel_size / 2
    #     img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
    #                   y_low - half_pixel_size, y_high + half_pixel_size]
    # except Exception:
    #     pass

    # tool_belt.create_image_figure(img_array, img_extent, clickHandler=None,
    #                     title=None, color_bar_label='Counts',
    #                     min_value=None, um_scaled=False)

    ############# Create csv filefor 2D image ##############
    # csv_filename = '{}_{}-us'.format(timestamp,int( CPG_pulse_dur/10**3))

    # tool_belt.save_image_data_csv(img_array, x_voltages, y_voltages, path,
    #                               csv_filename)
