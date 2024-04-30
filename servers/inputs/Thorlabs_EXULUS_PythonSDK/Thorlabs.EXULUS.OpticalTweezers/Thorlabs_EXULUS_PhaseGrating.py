import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from ctypes import *
import os
import time

try:
    from EXULUS_COMMAND_LIB import *
except OSError as ex:
    print("Warning:", ex)

try:
    from Thorlabs_EXULUS_CGHDisplay import *
except OSError as ex:
    print("Warning:", ex)

# Get the path to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load the DLL
dll_path = os.path.join(current_directory, "exulus_command_library.dll")
EXULUSLib = cdll.LoadLibrary(dll_path)

# SLM and Setup Settings
dimx = 1920
dimy = 1080
pp = 8
lda = 638

x, y = np.meshgrid(np.arange(dimx), np.arange(dimy))

# Grating Phase Pattern (XY Displacement)
g_x = 100
g_y = 0
phase_grating = np.mod(np.pi * g_x / dimx * x + np.pi * g_y / dimy * y, 2 * np.pi)
plt.imshow(phase_grating, cmap='gray', vmin=0, vmax=2 * np.pi)
plt.colorbar()
plt.show(block=False)  # Show image without blocking the script execution

# Convert phase to 16 bit and cast to c_uint16
phase_grating = (phase_grating / (2 * np.pi)) * 65535
phase_grating = phase_grating.astype(np.uint16)

# Define ctypes type for the phase grating array
phase_grating_type = c_uint16 * (dimx * dimy)

# Create a ctypes instance of the phase grating array
phase_grating_array = phase_grating_type(*phase_grating.flat)


def CommonFunc():
    hdl = CghDisplayCreateWindow(2, 1920, 1080, "SLM window")
    if hdl < 0:
        print("Create window failed")
        return -1
    else:
        print("Current screen is 2")

    result = CghDisplaySetWindowInfo(hdl, 1920, 1080, 1)
    if result < 0:
        print("Set Window Info failed")
    else:
        print("Set Window Info successfully")

    Image = [255] * 1920 * 1080
    dstr = (c_ubyte * len(Image))(*Image)
    result = CghDisplayShowWindow(hdl, dstr)
    if result < 0:
        print("Show failed")
    else:
        print("Show successfully")

    time.sleep(2)

    CghDisplayCloseWindow(hdl)


def main():
    print(" *** EXULUS device python example *** ")
    try:
        # Connect to EXULUS device
        devs = EXULUSListDevices()
        print(devs)
        if len(devs) <= 0:
            print('There are no devices connected')
            return
        else:
            EXULUS = devs[0]
            serial_number = EXULUS[0]
            hdl = EXULUSOpen(serial_number, 38400, 3)
            if hdl < 0:
                print("Connect", serial_number, "failed")
                return
            else:
                print("Connect", serial_number, "successfully")

        running = True
        while running:
            # Display the phase pattern on the SLM window
            hdl = CghDisplayCreateWindow(2, 1920, 1080, "SLM window")
            
            # Create a buffer from the phase_grating_array
            # buf = (c_uint16 * len(phase_grating_array))(*phase_grating_array)
            
            # Get the test pattern status (assuming it's a single byte value)
            test_pattern_status = c_ubyte()  # Example: 0x01 for On, 0x00 for Off
            
            # Set the test pattern status
            result = EXULUSSetTestPatternStatus(hdl, test_pattern_status)
        
            # Pass the buffer to the EXULUS function
            # result = EXULUSSetTestPatternStatus(hdl, buf)
            if result < 0:
                print("Failed to set test pattern status")
            
            # Check for keyboard input to stop
            if input("Press Enter to stop...") == "":
                running = False

    except IOError as ex:
        print("IOError:", ex)
    except OSError as ex:
        print("OSError:", ex)
    finally:
        # Close EXULUS device
        CghDisplayCloseWindow(hdl)
        EXULUSClose(hdl)
        print("*** End ***")


 
if __name__ == "__main__":
    main()
