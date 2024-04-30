import numpy as np
import os
import tqdm
import ctypes

try:
    from EXULUS_COMMAND_LIB import *
except OSError as ex:
    print("Warning:", ex)

# Get the path to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load the DLL
dll_path = os.path.join(current_directory, "exulus_command_library.dll")
EXULUSLib = ctypes.cdll.LoadLibrary(dll_path)

# Define GS algorithm
# def GS(source, target, retrieved_phase, it):
#     A = np.exp(retrieved_phase * 1j)
#     for i in tqdm.tqdm(range(0, it)):
#         B = np.abs(source) * np.exp(1j * np.angle(A))
#         C = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(B)))
#         D = np.abs(target) * np.exp(1j * np.angle(C))
#         A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(D)))
#     return np.angle(A)
def gaussian_beam(res1, res2, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, res1),np.linspace(-1, 1, res2))
    gauss = np.exp(-0.5*(1/sigma)*(x**2+y**2))
    return gauss

def GS(source, target, retrieved_phase, it):
    A = np.exp(retrieved_phase * 1j)
    for i in tqdm.tqdm(range(0, it)):
        B = np.abs(source) * np.exp(1j * np.angle(A))
        C = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(B)))
        D = np.abs(target) * np.exp(1j * np.angle(C))
        A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(D)))
        # Resize A to match the shape of target if necessary
        A = A[:target.shape[0], :target.shape[1]]
    return np.angle(A)

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

        # Define parameters for SLM
        dim_x = 1920
        dim_y = 1080

        # Generate Gaussian beam around the center of the SLM panel
        center_x = dim_x // 2
        center_y = dim_y // 2
        sigma = 0.2
        x, y = np.meshgrid(np.linspace(-1, 1, dim_x), np.linspace(-1, 1, dim_y))
        source = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)

        # Generate random initial phase
        phase_in = 2 * np.pi * np.random.rand(dim_x, dim_y)

        # Initialize an empty array for the target phase
        target_phase = np.zeros((dim_x, dim_y))

        # Main loop
        running = True
        while running:
            # Apply Gerchberg-Saxton algorithm to generate phase pattern
            retrieved_phase = GS(source, target_phase, phase_in, 50)

            # Convert phase array to bytes
            phase_bytes = retrieved_phase.astype(np.float32).tobytes()

            # Set phase stroke mode on SLM
            ret = EXULUSGetPhaseStrokeMode(hdl, phase_bytes)
            if ret != 0:
                print("Error setting phase stroke mode:", ret)

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