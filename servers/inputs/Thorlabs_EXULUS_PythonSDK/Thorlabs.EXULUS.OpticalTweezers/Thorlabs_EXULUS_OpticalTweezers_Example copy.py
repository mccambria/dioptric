import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from ctypes import *
import os
import tqdm

try:
    from EXULUS_COMMAND_LIB import *
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
g_x = 10
g_y = 0
phase_grating = np.mod(np.pi * g_x / dimx * x + np.pi * g_y / dimy * y, 2 * np.pi)
plt.imshow(phase_grating, cmap='gray', vmin=0, vmax=2 * np.pi)
plt.colorbar()
plt.show()

# Convert phase to 16 bit and cast to c_uint16
phase_grating = (phase_grating / (2 * np.pi)) * 65535
phase_grating = phase_grating.astype(np.uint16)

# Define ctypes type for the phase grating array
phase_grating_type = c_uint16 * (dimx * dimy)

# Create a ctypes instance of the phase grating array
phase_grating_array = phase_grating_type(*phase_grating.flat)


#region command for EXULUS
# Function declarations for EXULUS commands are here...

def plot(target_list, dim, spacing, origin):
    Array = np.zeros((dim*(spacing)+origin[0], dim*(spacing)+origin[1]))
    for i in target_list:
        Array[i[0], i[1]] = 1
#     plotArray(arr)
    i = 0
    j = 0
    x = []
    y = []
    while i<len(Array):
        while j<len(Array[0]):
            if Array[i][j] == True:
                x.append(j)
                y.append(i)
            j += 1
        i += 1
        j = 0
    ax.plot(x,y,'ro')
    plt.grid(True)
    plt.autoscale(False)
    plt.ylim(-1,len(Array))
    plt.xlim(-1,len(Array))

def gaussian_beam(res1, res2, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, res1),np.linspace(-1, 1, res2))
    gauss = np.exp(-0.5*(1/sigma)*(x**2+y**2))
    return gauss


def GS(source, target, retrived_phase, it):
    A = np.exp(retrived_phase*1j)
#     A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(target)))
#     plt.plot(),plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A)))), cmap = 'gray')
#     plt.title('T_in'), plt.xticks([]), plt.yticks([])
#     plt.show()
    
    for i in tqdm.tqdm(range(0,it)):
        B = np.abs(source)*np.exp(1j*np.angle(A))
        C = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(B)))
        D = np.abs(target)*np.exp(1j*np.angle(C))
        A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(D)))
        
    return np.angle(A)  

def convert_phase_to_bytes(phase):
    # Convert phase array to bytes
    phase_bytes = phase.astype(np.float32).tobytes()
    return phase_bytes

#------------------------ Configure Parameters -------------------------
# Configure slideshow (steer the laser beam from left to right and back):
# gratingPeriodMin = 8
# gratingPeriodMax = 64
# gratingPeriodStepSize = 4
dataDisplayDurationMilliSec = 1000  # duration of each data frame in ms
repeatSlideshow = -1  # <= 0 (e. g. -1) repeats until Python process gets killed
w = 2
dim_original = 5
spacing_pixel = 50
captured_prob = 0.5
origin = (500,500)
frame = 20

res1 = dim_original*spacing_pixel+origin[0]
res2 = dim_original*spacing_pixel+origin[1]


def main():
    print(" *** EXULUS device python example *** ")
    try:
        # Connect to EXULUS device
        devs = EXULUSListDevices()
        print(devs)
        if len(devs) <= 0:
            print('There is no devices connected')
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
 
        # Define parameters for optical tweezers
        source = gaussian_beam(res1, res2,0.2)
        phase_in = 2*np.pi*np.random.rand(res1 ,res2)
        A = source*np.exp(1j*phase_in)
        Array = np.zeros((dim_original*(spacing_pixel)+origin[0], dim_original*(spacing_pixel)+origin[1]))
        # Array = np.zeros(dim_x, dim_y)
        # Main loop
        running = True
        while running:
            # Generate optical tweezers array using GS algorithm
            phase = GS(source, Array, phase_in, 50)
            # Convert phase array to bytes
            phase_bytes = phase.astype(np.float32).tobytes()
           
            # Create a buffer for phase data
            phase_buffer = (c_ubyte * len(phase_bytes)).from_buffer_copy(phase_bytes)

            # Upload optical tweezers array to SLM
            ret = EXULUSGetPhaseStrokeMode(hdl, phase_buffer)
            if ret != 0:
                print("Error setting phase stroke mode:", ret)

            # EXULUSSetPhaseStrokeMode.argtypes = [c_int, POINTER(c_ubyte)]
            # EXULUSSetPhaseStrokeMode.restype = c_int
            # phase_ptr = cast(phase_bytes, POINTER(c_ubyte))
            # EXULUSSetPhaseStrokeMode(hdl, phase_ptr)

            # Check for keyboard input to stop
            if input("Press Enter to stop...") == "":
                running = False

    except Exception as ex:
        print("Warning:", ex)
    finally:
        # Close EXULUS device
        EXULUSClose(hdl)
        print("*** End ***")

if __name__ == "__main__":
    main()


# def main():
#     print(" *** EXULUS device python example *** ")
#     try:
#         devs = EXULUSListDevices()
#         print(devs)
#         if len(devs) <= 0:
#             print('There is no devices connected')
#         else:
#             EXULUS = devs[0]
#             serial_number = EXULUS[0]
#             hdl = EXULUSOpen(serial_number, 38400, 3)
#             if hdl < 0:
#                 print("Connect", serial_number, "failed")
#                 return
#             else:
#                 print("Connect", serial_number, "successfully")

#             # Define parameters for optical tweezers
#             dim_x = 100
#             dim_y = 100
#             spacing = 1
#             origin = (0, 0)
#             n_traps = 5
#             trap_centers = [(x, y) for x in np.linspace(origin[0], dim_x * spacing + origin[0], n_traps)
#                             for y in np.linspace(origin[1], dim_y * spacing + origin[1], n_traps)]
#             trap_strength = [1.0] * len(trap_centers)

#             # Generate optical tweezers array
#            or f in range(frame):

#          print(f)

#         sim = []
#         Array = np.zeros((dim_original*(spacing_pixel)+origin[0], dim_original*(spacing_pixel)+origin[1]))
#         for i in sim:
#             Array[i[0]-w:i[0]+w, i[1]-w:i[1]+w] = 1
                    
#         phase = GS(source, Array, phase_in, 50)
#         # phaseData = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(source*np.exp(phase*1j)))))

#         error, handle = EXULUSSetPhaseStrokeMode(phase)

     
#     # Make sure all data was loaded:
#     for handle in dataHandles:
#         error = slm.datahandleWaitFor(handle, slmdisplaysdk.State.ReadyToRender)
#         assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

#     print("100%")

#     except Exception as ex:
#         print("Warning:", ex)
#     print("*** End ***")

# if __name__ == "__main__":
#     main()
