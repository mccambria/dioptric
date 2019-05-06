# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:00:46 2019

This file takes in image_sample arrays and replots them with the units um.

@author: Aedan
"""

import json
import matplotlib.pyplot as plt
import numpy
#import os
from tkinter import Tk
from tkinter import filedialog
    


#import Utils.tool_belt as tool_belt

def recreate_scan_image(colorMap, fileType):
    """
    Creates a figure of a scan from the find_nvs function originally saved as a
    JSON .txt file. The created figure has axes plotted in microns and colorplot changes
    
    The function will open a window to select the file. This window may appear 
    behind Spyder, so just minimize Spyder to select a file.
    
    """
    print('Select file \n...')
    
    
    root = Tk()
    root.withdraw()
    root.focus_force()
    fileName = filedialog.askopenfilename(initialdir = "G:/Team Drives/Kolkowitz Lab Group/nvdata/image_sample",
                                          title = 'choose file to replot', filetypes = (("svg files","*.svg"),("all files","*.*")) ) 
    

    if fileName == '':
        print('No file selected')
    else: 
    
        fileNameBase = fileName[:-4]
        
        fileName = fileNameBase + '.txt'  
        print('File selected: ' + fileNameBase + '.svg')
    
        # Open the specified file
        with open(fileName) as json_file:
            
            # Load the data from the file
            data = json.load(json_file)
            
            # Read in the imgArray data into an array to be used as z-values. The last
            # line flips the matrix of values along the y axis (0) and then x axis (1)
            imgArray = []
            
            for line in data["img_array"]:
                imgArray.append(line)
                
            Z = numpy.flip(numpy.flip(imgArray, 0),1)
            
            # Read in the arrays of Center and Image Reoslution
            xyzCenters = data["coords"]
            imgResolution = data["num_steps"]
            
            # Read in the floating values for the scan ranges, centers, and resolution
            yScanRange = data["y_range"]
            yCenter = xyzCenters[1]
            yImgResolution = imgResolution
            
            xScanRange = data["x_range"]
            xCenter = xyzCenters[0]
            xImgResolution = imgResolution
        
        # Remove the file suffix on the file
        fileName = fileName[:-4]    
        
        # Define the scale from the voltso on the Galvo to microns
        # Currently using 35 microns per volt
        scale = 35
        
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
        fig, ax = plt.subplots(figsize=(8, 8 * aspRatio))
        
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
#        fig.set_tight_layout(True)
        fig.canvas.flush_events()
        
        # Save the file in the same file directory
        fig.savefig(fileName + '_replot.' + fileType)
        
if __name__ == "__main__":
  
    
    recreate_scan_image('inferno', 'png')
    