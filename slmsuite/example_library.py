import os, sys, time
# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), 'c:/Users/Saroj Chand/Documents/dioptric/servers/inputs'))

import numpy as np
import pandas as pd
import cv2
import h5py
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
# Generate a phase .gif
from IPython.display import Image
import imageio
import io 
mpl.rc('image', cmap='Blues')

def plot_intensity():
    # Data from the image
    data = {
        "Power": [1, 1.2, 1.4, 1.6, 2, 2.4, 3, 4, 6, 8],
        "all_array_spot Intensity": [73240, 74239, 75155, 76969, 80057, 84386, 89551, 103747, 153496, 296225],
        "oth order": [13010, 13154, 13213, 13333, 13638, 14087, 14599, 15705, 20399, 33304]
    }

    df = pd.DataFrame(data)
    # Compute the relative intensity
    df['Relative Intensity'] = df['all_array_spot Intensity'] / df['oth order']
    # Plot the data
    plt.figure(figsize=(10, 6))
    # Plot relative intensity
    plt.plot(df['Power'], df['Relative Intensity'], label='Relative Intensity (All Array / 0th Order)', marker='s', color='blue')
    # plt.plot(df['Power'], df['all_array_spot Intensity'], label='Array Spots Integrated Intensity', marker='o', color='orange')
    # plt.plot(df['Power'], df['oth order'], label='0th Order Intensity', marker='x', color='red')

    plt.xlabel('Power')
    plt.ylabel('Intensity')
    plt.title('Intensity vs Power')
    plt.legend()
    plt.grid(True)
    plt.show()
# plot_intensity()

def shift_phase(phase, shift_x, shift_y):
    for ind in range(phase.shape[0]):
        for jnd in range(phase.shape[1]):
            phase[ind, jnd] += np.dot((ind, jnd), (shift_x, shift_y))
    return phase

# Define the corresponding points in both coordinate systems
points_512 = np.array([[100, 100], [200, 100], [200, 200]], dtype='float32')  # Points in (512, 512) coordinate system
points_1480 = np.array([[300, 200], [600, 200], [600, 400]], dtype='float32')  # Corresponding points in (1480, 1020) coordinate system

# Compute the affine transformation matrix
M = cv2.getAffineTransform(points_512, points_1480)

# Save the affine transformation matrix to an .h5 file
with h5py.File('affine_transformation.h5', 'w') as f:
    f.create_dataset('affine_matrix', data=M)

# Function to transform a point using the affine transformation matrix
def transform_point_affine(point, M):
    point_homogeneous = np.array([point[0], point[1], 1]).reshape((3, 1))
    transformed_point_homogeneous = np.dot(M, point_homogeneous)
    return transformed_point_homogeneous.flatten()

# Example usage:
point_512 = np.array([150, 150])  # A point in the (512, 512) coordinate system
transformed_point = transform_point_affine(point_512, M)
print(f"Transformed point in the (1480, 1020) coordinate system: {transformed_point}")

# Load the affine transformation matrix from the .h5 file
with h5py.File('affine_transformation.h5', 'r') as f:
    M_loaded = f['affine_matrix'][:]

# Verify the loaded matrix is the same as the original matrix
print("Original matrix:\n", M)
print("Loaded matrix:\n", M_loaded)

# Transform a point using the loaded matrix to verify
transformed_point_loaded = transform_point_affine(point_512, M_loaded)
print(f"Transformed point using the loaded matrix in the (1480, 1020) coordinate system: {transformed_point_loaded}")
