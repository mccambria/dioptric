# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:59:44 2020

@author: agardill
"""
# %%
import copy
import time

import labrad
import matplotlib.pyplot as plt
import numpy

import majorroutines.image_sample as image_sample
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

# %%

reset_range = 1.5
image_range = 1.5
num_steps = int(225 * image_range)
num_steps_reset = int(75 * reset_range)
apd_indices = [0]
# red_reset_power = 0.8548
# red_pulse_power = 0.8548
# red_image_power = 0.568

green_reset_power = 0.6045
green_pulse_power = 0.635
green_image_power = 0.635

# %%


def plot_dif_fig(coords, x_voltages, range, dif_img_array, readout, title):
    x_coord = coords[0]
    half_x_range = range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    dif_img_array_kcps = (dif_img_array / 1000) / (readout / 10**9)

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [
        x_high + half_pixel_size,
        x_low - half_pixel_size,
        y_low - half_pixel_size,
        y_high + half_pixel_size,
    ]

    fig = tool_belt.create_image_figure(
        dif_img_array_kcps,
        img_extent,
        clickHandler=None,
        title=title,
        color_bar_label="Difference (kcps)",
        #                                        min_value = 0
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig


# %%
def main(
    cxn,
    base_sig,
    optimize_coords,
    center_coords,
    reset_coords,
    pulse_coords,
    pulse_time,
    init_time,
    init_color,
    pulse_color,
    readout_color,
    disable_optimize=False,
    wait_time=0,
):
    am_589_power = base_sig["am_589_power"]
    readout = base_sig["pulsed_SCC_readout_dur"]
    color_filter = base_sig["color_filter"]

    image_sig = copy.deepcopy(base_sig)
    image_sig["coords"] = center_coords
    image_sig["ao_515_pwr"] = green_image_power

    optimize_sig = copy.deepcopy(base_sig)
    optimize_sig["coords"] = optimize_coords
    optimize_sig["ao_515_pwr"] = green_image_power

    pulse_sig = copy.deepcopy(base_sig)
    pulse_sig["coords"] = pulse_coords
    pulse_sig["ao_515_pwr"] = green_pulse_power

    reset_sig = copy.deepcopy(base_sig)
    reset_sig["coords"] = reset_coords
    reset_sig["ao_515_pwr"] = green_reset_power

    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring["do_532_aom"]
    #    pulser_wiring_green = wiring['ao_515_laser']
    #    pulser_wiring_yellow = wiring['ao_589_aom']
    pulser_wiring_red = wiring["do_638_laser"]
    #    pulser_wiring_red = wiring['ao_638_laser']

    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params["515_laser_delay"]
    laser_589_delay = shared_params["589_aom_delay"]
    laser_638_delay = shared_params["638_DM_laser_delay"]
    #    laser_638_delay = shared_params['638_AM_laser_delay']

    if pulse_color == 532 or pulse_color == "515a":
        direct_wiring = pulser_wiring_green
        laser_delay = laser_515_delay
    elif pulse_color == 589:
        #        direct_wiring = pulser_wiring_yellow
        laser_delay = laser_589_delay
    elif pulse_color == 638:
        direct_wiring = pulser_wiring_red
        laser_delay = laser_638_delay

    #    optimize.main_xy_with_cxn(cxn, optimize_sig, apd_indices, 532, disable=disable_optimize)
    #
    #    adj_coords = (numpy.array(pulse_coords) + \
    #                  numpy.array(tool_belt.get_drift())).tolist()
    #    x_center, y_center, z_center = adj_coords
    #
    #    print('Resetting with {} nm light\n...'.format(init_color))
    #    _,_,_ = image_sample.main(reset_sig, reset_range, reset_range, num_steps_reset,
    #                      apd_indices, init_color,readout = init_time,  save_data=False, plot_data=False)
    #
    #    print('Waiting for {} s, during {} nm pulse'.format(pulse_time, pulse_color))
    #    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    #    # Use two methods to pulse the green light, depending on pulse length
    #    if pulse_time < 1:
    #        seq_args = [int(pulse_time*10**9), 0, 0.0, 0.0, 532]
    #        seq_args_string = tool_belt.encode_seq_args(seq_args)
    #        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)
    #    else:
    #        cxn.pulse_streamer.constant([], 0.0, 0.0)
    #        time.sleep(pulse_time)
    #    cxn.pulse_streamer.constant([], 0.0, 0.0)
    #
    #
    #    # collect an image
    #    print('Imaging {} nm light\n...'.format(readout_color))
    #    ref_img_array, x_voltages, y_voltages = image_sample.main(image_sig, image_range, image_range, num_steps,
    #                      apd_indices, readout_color,readout = readout, save_data=True, plot_data=True)

    #    optimize.main_xy_with_cxn(cxn, optimize_sig, apd_indices, 532, disable=disable_optimize)
    adj_coords = (
        numpy.array(pulse_coords) + numpy.array(tool_belt.get_drift())
    ).tolist()
    x_center, y_center, z_center = adj_coords

    print("Resetting with {} nm light\n...".format(init_color))
    _, _, _ = image_sample.main(
        reset_sig,
        reset_range,
        reset_range,
        num_steps_reset,
        apd_indices,
        init_color,
        readout=init_time,
        save_data=False,
        plot_data=False,
    )

    ## now pulse at the center of the scan for a short time
    print("Pulsing {} nm light for {} s".format(pulse_color, pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the light, depending on pulse length
    if pulse_time < 1.1:
        seq_args = [
            laser_delay,
            int(pulse_time * 10**9),
            am_589_power,
            green_pulse_power,
            pulse_color,
        ]
        print(seq_args)
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate("simple_pulse.py", 1, seq_args_string)
    else:
        if pulse_color == 532 or pulse_color == 638:
            cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
        elif pulse_color == 589:
            cxn.pulse_streamer.constant([], 0.0, am_589_power)
        elif pulse_color == "515a":
            cxn.pulse_streamer.constant([], green_pulse_power, 0)
        time.sleep(pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0)

    # collect an image under yellow after green pulse
    print("Imaging {} nm light\n...".format(readout_color))
    sig_img_array, x_voltages, y_voltages = image_sample.main(
        image_sig,
        image_range,
        image_range,
        num_steps,
        apd_indices,
        readout_color,
        readout=readout,
        save_data=True,
        plot_data=True,
    )

    # measure laser powers:
    (
        green_optical_power_pd,
        green_optical_power_mW,
        red_optical_power_pd,
        red_optical_power_mW,
        yellow_optical_power_pd,
        yellow_optical_power_mW,
    ) = tool_belt.measure_g_r_y_power(base_sig["am_589_power"], base_sig["nd_filter"])

    # Subtract and plot
    #    dif_img_array = sig_img_array - ref_img_array
    #
    #    if pulse_color == 532:
    #        pulse_power = green_optical_power_mW
    #    if pulse_color == 589:
    #        pulse_power = red_optical_power_mW
    #    if pulse_color == 638:
    #        pulse_power = yellow_optical_power_mW
    #    else :
    #        pulse_power = 0
    #
    #    title = 'Moving readout subtracted image, {} nm init, {} nm readout\n{} nm pulse {} s, {:.1f} mW'.format(init_color, readout_color, pulse_color, pulse_time, pulse_power)
    #    fig = plot_dif_fig(center_coords, x_voltages,image_range,  dif_img_array, readout, title )
    #
    #    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {
        "timestamp": timestamp,
        "init_color": init_color,
        "pulse_color": pulse_color,
        "readout_color": readout_color,
        "green_image_power": green_image_power,
        "green_image_power-units": "V",
        "green_reset_power": green_reset_power,
        "green_reset_power-units": "V",
        "green_pulse_power": green_pulse_power,
        "green_pulse_power-units": "V",
        "base_sig": base_sig,
        "base_sig-units": tool_belt.get_nv_sig_units(),
        "color_filter": color_filter,
        "optimize_coords": optimize_coords,
        "center_coords": center_coords,
        "pulse_coords": pulse_coords,
        "image_range": image_range,
        "image_range-units": "V",
        "num_steps": num_steps,
        "reset_range": reset_range,
        "reset_range-units": "V",
        "num_steps_reset": num_steps_reset,
        "init_time": init_time,
        "init_time-units": "ns",
        "pulse_time": float(pulse_time),
        "pulse_time-units": "s",
        "wait_time": int(wait_time),
        "wait_time-units": "s",
        "green_optical_power_pd": green_optical_power_pd,
        "green_optical_power_pd-units": "V",
        "green_optical_power_mW": green_optical_power_mW,
        "green_optical_power_mW-units": "mW",
        "red_optical_power_pd": red_optical_power_pd,
        "red_optical_power_pd-units": "V",
        "red_optical_power_mW": red_optical_power_mW,
        "red_optical_power_mW-units": "mW",
        "yellow_optical_power_pd": yellow_optical_power_pd,
        "yellow_optical_power_pd-units": "V",
        "yellow_optical_power_mW": yellow_optical_power_mW,
        "yellow_optical_power_mW-units": "mW",
        "readout": readout,
        "readout-units": "ns",
        "x_voltages": x_voltages.tolist(),
        "x_voltages-units": "V",
        "y_voltages": y_voltages.tolist(),
        "y_voltages-units": "V",
        #               'ref_img_array': ref_img_array.tolist(),
        #               'ref_img_array-units': 'counts',
        "sig_img_array": sig_img_array.tolist(),
        "sig_img_array-units": "counts",
        #               'dif_img_array': dif_img_array.tolist(),
        #               'dif_img_array-units': 'counts'
    }

    filePath = tool_belt.get_file_path("image_sample", timestamp, base_sig["name"])
    tool_belt.save_raw_data(rawData, filePath + "_dif")
    #
    #    tool_belt.save_figure(fig, filePath + '_dif')

    return


# %%
if __name__ == "__main__":
    sample_name = "goeppert-mayer"

    base_sig = {
        "coords": [],
        "name": "{}".format(sample_name),
        "expected_count_rate": 55,
        "nd_filter": "nd_0",
        #            'color_filter': '635-715 bp',
        "color_filter": "715 lp",
        "pulsed_readout_dur": 300,
        "pulsed_SCC_readout_dur": 4 * 10**7,
        "am_589_power": 0.3,
        "pulsed_initial_ion_dur": 25 * 10**3,
        "pulsed_shelf_dur": 200,
        "am_589_shelf_power": 0.35,
        "pulsed_ionization_dur": 10**3,
        "cobalt_638_power": 130,
        "pulsed_reionization_dur": 100 * 10**3,
        "cobalt_532_power": 10,
        "ao_515_pwr": 0.65,
        "magnet_angle": 0,
        "resonance_LOW": 2.7,
        "rabi_LOW": 146.2,
        "uwave_power_LOW": 9.0,
        "resonance_HIGH": 2.9774,
        "rabi_HIGH": 95.2,
        "uwave_power_HIGH": 10.0,
    }
    expected_count_list = [36, 40, 35, 47, 52, 33, 40]  # 3/1/21
    start_coords_list = [
        [0.032, 0.147, 5.25],
        [0.068, 0.132, 5.28],
        [-0.007, 0.173, 5.19],
        [0.221, -0.163, 5.21],
        [0.207, -0.261, 5.23],
        [0.100, -0.212, 5.27],
        [0.118, -0.110, 5.20],
        [-0.390, -0.376, 5.26],
        [-0.362, -0.341, 5.20],
        [-0.237, -0.293, 5.22],
        [-0.384, -0.249, 5.20],
        [0.186, 0.247, 5.23],
        [0.226, 0.269, 5.22],
        [0.231, 0.319, 5.19],
        [0.372, 0.212, 5.18],
    ]

    init_time = 10**7

    init_color = 638
    pulse_color = 532
    readout_color = 589

    center_coords = [0.231, 0.319, 5.19]
    reset_coords = [0.231, 0.319, 5.19]
    optimize_coords = [0.231, 0.319, 5.19]
    #    center_coords = pulse_coords

    with labrad.connect() as cxn:
        pulse_time = 100
        pulse_coords = [0.231, 0.319, 5.19]
        main(
            cxn,
            base_sig,
            optimize_coords,
            center_coords,
            reset_coords,
            pulse_coords,
            pulse_time,
            init_time,
            init_color,
            pulse_color,
            readout_color,
        )
