# -*- coding: utf-8 -*-
"""
These objects are useful for sweep routines. For example we may want to find
where the NVs on a diamond sample are, we will sweep out an area of the
diamond, collecting fluorescence at each point in the sweep. Our sweeps will
be conducted in a winding pattern (first to last in x, then advance in y, then
first to last in x, then advance y...). These functions abstract most of the
work necessary to facilitate this.

Created on Tue Jan  9 19:25:34 2019

@author: mccambria
"""

# User modules

# Library modules
import numpy
from enum import Enum
from enum import auto


def calc_voltages(resolution, xLow, yLow, xNumSteps, yNumSteps):
    """
    Calculate the ndarray of sweep voltages to be passed to the galvo.

    Params:
        resolution: float
            Volts per step between samples
        xLow: float
            x offset voltage to align the low voltage of the sweep
        yLow: float
            y offset voltage to align the low voltage of the sweep
        xNumSteps: int
            Number of steps in the x direction
        yNumSteps: int
            Number of steps in the y direction

    Returns:
        numpy.ndarray: the calculated ndarray
    """

    # Set up vectors for the number of samples in each direction
    # [0, 1, 2, ... length - 1]
    xSteps = numpy.arange(xNumSteps)
    ySteps = numpy.arange(yNumSteps)

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions are
    # linear in these voltages. For a small range, however, we don't really
    # care.
    xVoltages = (resolution * xSteps) + xLow
    yVoltages = (resolution * ySteps) + yLow

    # Return the array of voltages to apply to the galvo
    return winding_cartesian_product(xVoltages, yVoltages)


def winding_cartesian_product(xVector, yVector):
    """
    For two input vectors (1D ndarrays) of lengths n and m, returns a
    ndarray of length n * m representing every ordered pair of elements in
    a winding pattern (first to last in x, then advance in y, then
    first to last in x, then advance y...). Copy params determine the
    number of copies of each output row.

    Example:
        winding_cartesian_product([1, 2, 3], [4, 5, 6]) returns
        [[1, 2, 3, 3, 2, 1, 1, 2, 3],
         [4, 4, 4, 5, 5, 5, 6, 6, 6]]

    Params:
        xVector: numpy.ndarray
            A 1xn numpy ndarray of the x values
        yVector: numpy.ndarray
            A 1xn numpy ndarray of the y values

    Returns:
        numpy.ndarray: the calculated ndarray
    """

    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for
    # cartesian_product([1, 2, 3], [4, 5, 6])

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    xInter = numpy.concatenate((xVector, numpy.flipud(xVector)))
    ySize = yVector.size
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if ySize % 2 == 0:  # Even x size
        xVals = numpy.tile(xInter, int(ySize/2))
    else:  # Odd x size
        xVals = numpy.tile(xInter, int(numpy.floor(ySize/2)))
        xVals = numpy.concatenate((xVals, xVector))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    yVals = numpy.repeat(yVector, xVector.size)

    # Stack the input vectors
    return numpy.stack((xVals, yVals))


class SweepStartingPosition(Enum):
    TOPLEFT = auto()
    TOPRIGHT = auto()
    BOTTOMLEFT = auto()
    BOTTOMRIGHT = auto()


def populate_img_array(valsToAdd, imgArray, writePos,
                       startingPos=SweepStartingPosition.BOTTOMRIGHT):
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

    # For now the code supports winding from lowest x/y voltage to highest
    # x/y voltage. On the galvo, higher y voltage => down and higher x
    # voltage => left. So the currently supported voltage winding maps to
    # image winding from the bottom right corner to the top left corner.
    if startingPos is SweepStartingPosition.BOTTOMRIGHT:
        if len(writePos) == 0:
            writePos[:] = [xDim, yDim - 1]
    else:
        raise NotImplementedError()
        return

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
