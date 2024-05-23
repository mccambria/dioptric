import os, sys, time
# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), 'c:/Users/Saroj Chand/Documents/dioptric/servers/inputs'))

import numpy as np
import pandas as pd
import cv2
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

from slmsuite.holography import analysis, toolbox
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram

# funtions
# region "plot_phase" function
def plot_phase(phase, title="", zoom=True):
    # One plot if no camera; two otherwise.
    _, axs = plt.subplots(1, 2 - (cam is None), figsize=(24,6))

    if cam is None:
        axs = [axs]

    # Plot the phase.
    axs[0].set_title("SLM Phase")
    im = axs[0].imshow(
        np.mod(phase, 2*np.pi),
        vmin=0,
        vmax=2*np.pi,
        interpolation="none",
        cmap="twilight"
    )
    plt.colorbar(im, ax=axs[0])

    # Grab an image of the resulting pattern and plot.
    slm.write(phase, settle=True)
    img = cam.get_image()

    axs[1].set_title("Camera Result")
    axs[1].imshow(img)
    if zoom:
        xlim = axs[1].get_xlim()
        ylim = axs[1].get_ylim()
        axs[1].set_xlim([xlim[0] * .7 + xlim[1] * .3, xlim[0] * .3 + xlim[1] * .7])
        axs[1].set_ylim([ylim[0] * .7 + ylim[1] * .3, ylim[0] * .3 + ylim[1] * .7])

    # Make a title, if given.
    plt.suptitle(title)
    plt.show()

# region "blaze" function
def blaze():
    # Get .2 degrees in normalized units.
    vector_deg = (.2, .2)
    vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")

    # Get the phase for the new vector
    blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)

    plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))

# region "calibration"
def fourier_calibration():
    cam.set_exposure(.006)               # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[25, 16],           # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[30, 40],           # Pitch of the calibration grid (x, y) [knm]
        plot=True
    )
    cam.set_exposure(.0001)
    #save calibation
    calibration_file = fs.save_fourier_calibration(path=r"C:\Users\Saroj Chand\Documents\fourier_calibration")
    print("Fouri er calibration saved to:", calibration_file)

def test_wavefront_calibration():
    cam.set_exposure(.001)
    movie = fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(.25, 0),
        field_point_units="freq",
        superpixel_size=60,
        test_superpixel=(16, 16),           # Testing mode
        autoexposure=False,
        plot=3                              # Special mode to generate a phase .gif
    )
    imageio.mimsave('wavefront.gif', movie)
    Image(filename="wavefront.gif")

def wavefront_calibration():
    cam.set_exposure(.001)
    fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(.25, 0),
        field_point_units="freq",
        superpixel_size=40,
        autoexposure=False
    )
    #save calibation
    calibration_file = fs.save_wavefront_calibration(path=r"C:\Users\Saroj Chand\Documents\wavefront_calibration")
    print("Fourier calibration saved to:", calibration_file)
    
# region "load calibration" 
def load_fourier_calibration():
    calibration_file_path = r"C:\Users\Saroj Chand\Documents\fourier_calibration\26438-SLM-fourier-calibration_00007.h5"
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)

def load_wavefront_calibration():
    calibration_file_path = r"C:\Users\Saroj Chand\Documents\wavefront_calibration\26438-SLM-wavefront-calibration_00004.h5"
    fs.load_wavefront_calibration(calibration_file_path)
    print("Wavefront calibration loaded from:", calibration_file_path)
    
# region "cam_plot" function 
def cam_plot():
    cam.set_exposure(.0001)
    img = cam.get_image()

    # Plot the result
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.show()
    
# region "evaluate" function 
def evaluate_uniformity(vectors=None, size=25):
    # Set exposure and capture image
    cam.set_exposure(0.00001)
    img = cam.get_image()

    # Extract subimages
    if vectors is None:
        subimages = analysis.take(img, vectors=None, size=size)
    else:
        subimages = analysis.take(img, vectors=vectors, size=size)

    # Plot subimages
    analysis.take_plot(subimages)

    # Normalize subimages and compute powers
    powers = analysis.image_normalization(subimages)

    # Plot histogram of powers
    plt.hist(powers / np.mean(powers))
    plt.show()

    # return subimages, powers

# region "square" function 
def square_array():
    xlist = np.arange(550, 1150, 25)                      # Get the coordinates for one edge
    ylist = np.arange(240, 840, 25) 
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))      # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    hologram.plot_nearfield(title="Padded", padded=True)
    hologram.plot_nearfield(title="Unpadded")
    phase = hologram.extract_phase()
    for ind in range(phase.shape[0]):
        for jnd in range(phase.shape[1]):
            phase[ind, jnd] += np.dot((ind, jnd), (0.03, 0.03))
    slm.write(phase, settle=True)
    movie = cam_plot()
    # evaluate_uniformity(vectors=square)
    # Hone the result with experimental feedback.
    # hologram.optimize(
    #     'WGS-Kim',
    #     maxiter=20,
    #     feedback='experimental_spot',
    #     stat_groups=['computational_spot', 'experimental_spot'],
    #     fixed_phase=False
    # )
    # phase = hologram.extract_phase()
    # slm.write(phase, settle=True)
    # cam_plot()
    # evaluate_uniformity(vectors=square)

# region "circle" function 
def circle_pattern():
    cam.set_exposure(0.001)
   # Define parameters for the circle
    center = (850, 540)  # Center of the circle
    radius = 200  # Radius of the circle

    # Generate points within the circle using polar coordinates
    num_points = 30  # Number of points to generate
    theta = np.linspace(0, 2*np.pi, num_points)  # Angle values
    x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    # Convert to grid format if needed
    circle = np.vstack((x_circle, y_circle))

    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=circle, basis='ij', cameraslm=fs)

    # # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)


    # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)

def circles():
    cam.set_exposure(0.001)
   # Define parameters for the circle
    center = (580, 400)  # Center of the circle
    # Generate circles with radii ranging from 10 to 200
    radii = np.linspace(50, 200, num=4)  # Adjust the number of circles as needed
    # Generate points for each circle and create the hologram
    circle_points = []
    for radius in radii:
        num_points = int(2 * np.pi * radius / 60)  # Adjust the number of points based on the radius

        # Generate points within the circle using polar coordinates
        theta = np.linspace(0, 2*np.pi, num_points)  # Angle values
        x_circle = center[0] + radius * np.cos(theta)  # X coordinates
        y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

        # Convert to grid format for the current circle
        circle = np.vstack((x_circle, y_circle))
        
        circle_points.append(circle)

    # Combine the points of all circles
    circles = np.concatenate(circle_points, axis=1)
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=circles, basis='ij', cameraslm=fs)

    # # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)


    # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)
# region "scatter" function 
def scatter_pattern():
    lloyds_points = toolbox.lloyds_points(
        grid=tuple(int(s/5) for s in fs.cam.shape), 
        n_points=100, 
        iterations=40
        ) * 5
    
    hologram = SpotHologram((2048, 2048), lloyds_points, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    # # Hone the result with experimental feedback.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='experimental_spot',
        stat_groups=['computational_spot', 'experimental_spot'],
        fixed_phase=False
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

# region "smiley" function 

def smiley_pattern():
  # Define points for the smiley face
    x_eyes = [1100, 1200]  # X coordinates for the eyes
    y_eyes = [400, 400]     # Y coordinates for the eyes

    # Define points for the mouth (a semi-circle)
    theta = np.linspace(0, np.pi, 15)  # Angle values for the semi-circle
    mouth_radius = 150  # Radius of the mouth
    x_mouth = 1150 + mouth_radius * np.cos(theta)  # X coordinates for the mouth
    y_mouth = 500 + mouth_radius * np.sin(theta)   # Y coordinates for the mouth

    # Combine all points into a single array
    smiley = np.vstack((np.concatenate((x_eyes, x_mouth)), np.concatenate((y_eyes, y_mouth))))
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=smiley, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

# region "UCB"  function 
def UCB_pattern():
    # Define coordinates for each letter in "UCB"
    letters = {
        'U': [(700, 400), (650, 500), (650, 600), (700, 700), (750, 600), (750, 500)],
        'C': [(800, 500), (800, 600), (850, 600), (900, 550), (850, 500), (900, 500)],
        'B': [(950, 500), (950, 600), (1000, 600), (1050, 550), (1000, 500), (1050, 500)],
    }

    # Combine coordinates for "UCB"
    ucb = np.vstack([letters[letter] for letter in "UCB"]).T

    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=ucb, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

def pattern_from_image():
        # Load the image of the letters "UCB"
    image = cv2.imread("cgn30eDI_400x400.png", cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image to obtain binary representation
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Display the thresholded image for debugging
    cv2.imshow("Thresholded Image", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Find contours of the letters in the image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Display the contours for debugging
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
          # Extract the coordinates of the contours
    ucb_coords = []
    for contour in contours:
        for point in contour:
            ucb_coords.append([point[0][0], point[0][1]])

    # Print out ucb_coords for debugging
    print("ucb_coords:", ucb_coords)

    # Convert ucb_coords to numpy array and transpose it
    ucb_coords = np.vstack(ucb_coords).T
    
    # Print out ucb_coords after conversion for debugging
    print("ucb_coords after conversion:", ucb_coords)

    # Transpose the coordinates array to match the expected shape (2, N)
    ucb = ucb_coords
    
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=ucb, basis='ij', cameraslm=fs)
    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()

# integrate Intensity
def integrate_intensity():
    xlist = np.arange(650, 750, 25)                      # Get the coordinates for one edge
    ylist = np.arange(340, 440, 25) 
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))      # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )
    hologram.plot_nearfield(title="Padded", padded=True)
    hologram.plot_nearfield(title="Unpadded")
    phase = hologram.extract_phase()
    # for ind in range(phase.shape[0]):
    #     for jnd in range(phase.shape[1]):
    #         phase[ind, jnd] *= np.dot((ind, jnd), (0.5, 0))
    # print(phase)
    # phase *= np.exp(1j * np.dot(np.meshgrid(phase.shape), (10,15))), 855,502.5
    slm.write(phase, settle=True)
    intensity_map = cam_plot()
    cam.set_exposure(.0001)
    img = cam.get_image()
    # Define the region of interest (ROI) around the center spot and compute intensity
    x, y = 700, 390  # Center spot coordinates, adjust as necessary
    roi_size = 110  # Size of the region around the center spot to analyze
    center_intensity = img[y-roi_size:y+roi_size, x-roi_size:x+roi_size].sum()
    roi_size = 50
    x, y = 830, 540
    # Compute intensities for each spot in the grid
    total_intensity = img[y-roi_size:y+roi_size, x-roi_size:x+roi_size].sum()
    # Print or plot the intensities for analysis
    print("Center spot intensity:", center_intensity)
    print(f"all intensity: {total_intensity}")

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

def save_initial_phase():
    xlist = np.arange(550, 1150, 25)  # Get the coordinates for one edge
    ylist = np.arange(240, 840, 25)
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))  # Make an array of points in a grid
    hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    # Precondition computationally
    hologram.optimize(
        'WGS-Kim',
        maxiter=20,
        feedback='computational_spot',
        stat_groups=['computational_spot']
    )

    initial_phase = hologram.extract_phase()
    np.save('initial_phase.npy', initial_phase)  # Save the phase to a file
    print("Initial phase saved.")

def animate_wavefront_shifts():
    initial_phase = np.load('initial_phase.npy')  # Load the saved phase

    frames = []
    shifts = np.linspace(0, 0.3, num=10)  # Define the range and number of steps for the shifts

    # Define the square path
    path = []
    for shift in shifts:
        path.append((shift, 0))  # Right
    for shift in shifts:
        path.append((shifts[-1], shift))  # Up
    for shift in shifts[::-1]:
        path.append((shift, shifts[-1]))  # Left
    for shift in shifts[::-1]:
        path.append((0, shift))  # Down

    for shift_x, shift_y in path:
        shifted_phase = shift_phase(np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y)
        slm.write(shifted_phase, settle=True)
        cam.set_exposure(0.0001)
        im = cam.get_image()
        if im is not None:
            plt.imshow(im)  # Use custom colormap
            plt.draw()
            # Save the current figure to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = imageio.imread(buffer)
            frames.append(image)
            buffer.close()
            plt.pause(0.01)  # Small pause to update the plot

    # Save frames as a GIF
    imageio.mimsave('wavefront_shift_animation.gif', frames, fps=5)
    print("Animation saved as wavefront_shift_animation.gif")

    # # Ensure frames are in the correct format for imageio
    # frames = [np.array(frame, dtype=np.uint8) for frame in frames]
    # imageio.mimsave('wavefront_shift_animation_1.gif', frames, fps=30)

# region run funtions
try:
    slm = ThorSLM(serialNumber='00429430')
    cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)
    # blaze()
    # fourier_calibration()
    load_fourier_calibration()
    # test_wavefront_calibration()
    # wavefront_calibration()
    load_wavefront_calibration()
    # fs.process_wavefront_calibration(r2_threshold=.9, smooth=True, plot=True)
    # square_array()
    save_initial_phase()
    animate_wavefront_shifts()
    # circles()
    # circle_pattern()
    # smiley()pip i
    # scatter_pattern()
    # UCB_pattern()
    # pattern_from_image()
    # cam_plot()
    # integrate_intensity()

finally:
    print("Closing")
    # After you're done using the camera
    cam.close()  # Add this line
    # Then close the SDK
    ThorCam.close_sdk()

    # slm.close()
# endregions