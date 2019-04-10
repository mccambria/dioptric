# -*- coding: utf-8 -*-
"""
Scan the galvos over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on Tue Apr  9 15:18:53 2019

@author: Matt
"""

import numpy
import utils.tool_belt as tool_belt
import time


def populate_img_array(valsToAdd, imgArray, writePos):
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
            The last x, y write position on the image array. (-1, 0) to
            start a new image from the top left corner.
        startingPos: SweepStartingPosition
            Sweep starting position of the winding pattern

    Returns:
        numpy.ndarray: The updated imgArray
        tuple(int): The last x, y write position on the image array
    """

    yDim = imgArray.shape[0]
    xDim = imgArray.shape[1]

    xPos = writePos[0]
    yPos = writePos[1]

    # Figure out what direction we're heading
    headingLeft = ((yDim - 1 - yPos) % 2 == 0)

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if (xPos == 0):
                yPos = yPos - 1
                imgArray[yPos, xPos] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos] = val
        else:
            # Determine if we're at the right x edge
            if (xPos == xDim - 1):
                yPos = yPos - 1
                imgArray[yPos, xPos] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos + 1
                imgArray[yPos, xPos] = val
    writePos[:] = [xPos, yPos]


def on_click_image(event):
    """
    Click handler for images. Prints the click coordinates to the console.

    Params:
        event: dictionary
            Dictionary containing event details
    """

    try:
        print('\n    xCenter = %.3f\n    yCenter = %.3f' %
              (event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass


def main(name, x_center, y_center, z_center, x_range, y_range,
         num_steps, readout, apd_index, continuous=False):

    # %% Some initial calculations

    cxn = tool_belt.get_cxn()

    if x_range != y_range:
        raise RuntimeError('x and y resolutions must match for now.')

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    period = readout + numpy.int64(500000)

    # %% Set up the galvo

    return_vals = cxn.galvo.load_scan(x_center, y_center, x_range, y_range,
                                      num_steps, period)

    x_num_steps, x_low, x_high, y_num_steps, y_low, y_high, pixel_size = return_vals

    total_num_samples = x_num_steps * y_num_steps

    # %% Set the piezo

    cxn.objective_piezo.write_voltage(z_center)

    # %% Set up the APD

    cxn.apd.load_stream_reader(apd_index, period, total_num_samples)

    # %% Set up the image display

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, y_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []

    # For the image extent, we need to bump out the min/max x/y by half the
    # pixel size in each direction so that the center of each pixel properly
    # lies at its x/y voltages.
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    fig = tool_belt.create_image_figure(img_array, img_extent,
                                        clickHandler=on_click_image)

    # %% Run the PulseStreamer

    cxn.pulse_streamer.stream_immediate('simple_readout.py', [period, readout])

    # %% Collect the data

    timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
    timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    while num_read_so_far < total_num_samples:

        if time.time() > timeout_inst:
            raise Warning('scan_sample timed out before all '
                          'samples were collected.')
            break

        # Read the samples and update the image
        new_samples = cxn.read_stream(apd_index)
        populate_img_array(new_samples, img_array, img_write_pos)
        tool_belt.update_image_figure(fig, img_array)
        num_read_so_far += len(new_samples)

    # %% Clean up

    # Close tasks
    cxn.galvo.close_task()
    cxn.apd.close_task()

    # Return to center
    cxn.galvo.write(x_center, y_center)
    cxn.objective_piezo.write_voltage(z_center)

    # %% Save the data

    timeStamp = tool_belt.get_time_stamp()

    rawData = {'timeStamp': timeStamp,
               'name': name,
               'xyz_centers': [x_center, y_center, z_center],
               'x_range': x_range,
               'y_range': y_range,
               'num_steps': num_steps,
               'readout': int(readout),
               'resolution': [x_num_steps, x_num_steps],
               'img_array': img_array.astype(int).tolist()}

    filePath = tool_belt.get_file_path('scan_sample', timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
