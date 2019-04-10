# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:00:46 2019

@author: Aedan
"""

import json
import matplotlib.pyplot as plt
import numpy

#import Utils.tool_belt as tool_belt

def recreate_scan_image(fileName, colorMap, fileType):
    """
    Creates a figure of a scan from the find_nvs function originally saved as a
    JSON .txt file. The created figure has axes plotted in microns and colorplot changes

    Params:
        fileName: string
            Filename of the image to recreate. Include everything up to .txt
        colorMap: string
            Specifies the colormapping to use with figure. For colormaps, see 
            matplotlib.org/examples/color/colormaps_reference.html
        fileType: string
            Specify the file type to save as. Examples: png, pdf, svg. 
            Exclude the period.
    Returns:
        matplotlib.figure.Figure? (I think)
    """

        # Open the specified file
    with open(fileName + '.txt') as json_file:
        
        # Load the data from the file
        data = json.load(json_file)
        
        # Read in the imgArray data into an array to be used as z-values. The last
        # line flips the matrix of values along the y axis (0) and then x axis (1)
        imgArray = []
        
        for line in data["imgArray"]:
            imgArray.append(line)
            
        Z = numpy.flip(numpy.flip(imgArray, 0),1)
        
        # Read in the arrays of Center and Image Reoslution
        xyzCenters = data["xyzCenters"]
        imgResolution = data["imgResolution"]
        
        # Read in the floating values for the scan ranges, centers, and resolution
        yScanRange = data["yScanRange"]
        yCenter = xyzCenters[1]
        yImgResolution = imgResolution[1]
        
        xScanRange = data["xScanRange"]
        xCenter = xyzCenters[0]
        xImgResolution = imgResolution[0]
        
        # Define the scale from the voltso on the Galvo to microns
        # Currently using 140 microns per volt
        scale = 1
        
        # Calculate various values pertaining to the positions in the image
        xScanCenterPlusMinus = xScanRange / 2
        xImgStepSize = xScanRange / xImgResolution
        xMin = xCenter - xScanCenterPlusMinus
        
        yScanCenterPlusMinus = yScanRange / 2
        yImgStepSize = yScanRange / yImgResolution
        yMin = yCenter - yScanCenterPlusMinus
        
        # Generate the X and Y arrays for positions. The position refers to the 
        # bottom left corner of a pixel
        X = []
        X.append(xMin)
        i = 1
        while i < (xImgResolution + 1):
            xNextPoint = X[i - 1] + xImgStepSize
            X.append(xNextPoint)
            i += 1
            
        Y = []
        Y.append(yMin)
        i = 1
        while i < (yImgResolution + 1):
            yNextPoint = Y[i - 1] + yImgStepSize
            Y.append(yNextPoint)
            i += 1
            
        # Calculate the aspect ratio between y and x , to be used in the figsize
        aspRatio = yImgResolution / xImgResolution
        
        # Create the figure, specifying only one plot. x and y label inputs are self-
        # explanatory. cmap allows a choice of color mapping.
        fig, ax = plt.subplots(figsize=(20, 20 * aspRatio))
        
        # Specifying various parameters of the plot, add or comment out as needed:
        # x and y axes labels
        # add title
        # add colorbar
        
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
#        plt.set_title('WeS2')
#        plt.colorbar()
        
        # Telling matplotlib what to plot, and what color map to include
        img = ax.pcolor(scale * numpy.array(X), scale * numpy.array(Y), Z, cmap=colorMap)

        fig.canvas.draw()
        fig.set_tight_layout(True)
        fig.canvas.flush_events()
        
        # Save the file in the same file directory
        fig.savefig(fileName + 'replot.' + fileType)
        
if __name__ == "__main__":
    recreate_scan_image('2019-03-08_15-53-35_WSe2', 'inferno', 'png')
    