import matplotlib.pyplot as plt
import numpy as np

from slmsuite.hardware.slms.thorlabs import ThorSLM
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def shift_phase(phase, shift_x, shift_y):
    """
    Shift the phase by adding a phase gradient.

    Parameters:
        phase (np.ndarray): The current phase array.
        shift_x (float): The shift in the x direction.
        shift_y (float): The shift in the y direction.

    Returns:
        np.ndarray: The shifted phase array.
    """
    y_indices, x_indices = np.indices(phase.shape)
    phase_shift = shift_x * x_indices + shift_y * y_indices
    shifted_phase = phase + phase_shift
    return shifted_phase


def scanning_widefield_image(slm, initial_phase, shift_range=(-1.0, 1.0), num_steps=10):
    """
    Collect counts for different phase shifts using SLM.

    Parameters:
        slm (ThorSLM): The SLM object.
        initial_phase (np.ndarray): The initial phase array.
        shift_range (tuple): Range of phase shifts.
        num_steps (int): Number of steps for shifting.

    Returns:
        np.ndarray: Phase shifts.
        np.ndarray: Corresponding counts.
    """
    shifts = np.linspace(shift_range[0], shift_range[1], num_steps)
    img_array_list = []

    for shift in shifts:
        shifted_phase = shift_phase(initial_phase, shift, 0)
        slm.write(shifted_phase)

        # Capture the image
        img_array = capture_image()
        img_array_list.append(img_array)

    return shifts, img_array_list


def capture_image(num_reps=1, display_image=False):
    """
    Capture an image using the SLM and camera.

    Parameters:
        num_reps (int): Number of repetitions for averaging the image.
        display_image (bool): If True, display the captured image.

    Returns:
        np.ndarray: Averaged image array.
    """
    tb.reset_cfm()
    laser_key = VirtualLaserKey.WIDEFIELD_IMAGING
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["name"]
    readout_duration = laser_dict["duration"]

    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    # Prepare sequence arguments
    seq_args = [readout_duration, readout_laser]

    # Load the pulse sequence
    pulse_gen.stream_load(
        "simple_readout-widefield-scanning.py", tb.encode_seq_args(seq_args), num_reps
    )

    camera.arm()
    img_array_list = []

    def rep_fn(rep_ind=1):
        img_str = camera.read()
        sub_img_array, _ = widefield.img_str_to_array(img_str)
        img_array_list.append(sub_img_array)

    widefield.rep_loop(num_reps, rep_fn)
    camera.disarm()

    img_array = np.mean(img_array_list, axis=0)

    if display_image:
        fig, ax = plt.subplots()
        kpl.imshow(ax, img_array, title="Widefield Scanning Image", cbar_label="ADUs")
        plt.show()

    return img_array


def save_data(
    fig, img_array, nv_sig, shifts, num_reps, readout_ms, title, save_dict=None
):
    """
    Save the captured image and metadata.

    Parameters:
        fig (matplotlib.figure.Figure): The figure containing the image.
        img_array (np.ndarray): The image data to save.
        nv_sig (NVSig): NV signal information used for positioning and imaging.
        shifts (np.ndarray): The applied phase shifts.
        num_reps (int): Number of repetitions.
        readout_ms (float): Readout duration in milliseconds.
        title (str): Title for the saved data.
        save_dict (dict): Additional metadata to save.

    Returns:
        None
    """
    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "caller_fn_name": "scanning_widefield_image",
        "nv_sig": nv_sig,
        "num_reps": num_reps,
        "readout": readout_ms,
        "readout-units": "ms",
        "title": title,
        "img_array": img_array,
        "img_array-units": "counts",
        "shifts": shifts,
    }

    if save_dict is not None:
        raw_data.update(save_dict)

    nv_name = nv_sig.name if nv_sig else "default_nv"
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_figure(fig, file_path)
    dm.save_raw_data(raw_data, file_path, keys_to_compress=["img_array"])


def main(
    slm,
    nv_sig=None,
    initial_phase=None,
    shift_range=(-1.0, 1.0),
    num_steps=10,
    num_reps=20,
):
    """
    Perform the SLM-based scanning and save the results.

    Parameters:
        slm (ThorSLM): The SLM object.
        nv_sig (NVSig, optional): NV signal information used for positioning and imaging.
        initial_phase (np.ndarray): Initial phase for the SLM.
        shift_range (tuple): Range for the phase shifts.
        num_steps (int): Number of phase shift steps.
        num_reps (int): Number of repetitions for averaging the image.

    Returns:
        np.ndarray: The final averaged image.
    """
    # If initial phase is not provided, create a zero-phase array based on the SLM shape
    if initial_phase is None:
        initial_phase = np.zeros(slm.shape)

    # Generate phase shifts and collect images
    shifts, img_array_list = scanning_widefield_image(
        slm, initial_phase, shift_range, num_steps
    )

    # Average the image arrays
    avg_img_array = np.mean(img_array_list, axis=0)

    # Metadata setup
    readout_ms = 100  # Example readout duration
    title = "Averaged Scanning Image"

    # Create the figure
    fig, ax = plt.subplots()
    kpl.imshow(ax, avg_img_array, title=title, cbar_label="ADUs")
    plt.show()

    # Save data
    save_dict = {"shift_range": shift_range, "num_steps": num_steps}
    save_data(
        fig, avg_img_array, nv_sig, shifts, num_reps, readout_ms, title, save_dict
    )

    return avg_img_array


if __name__ == "__main__":
    kpl.init_kplotlib()
    nv_sig = None  # Replace with actual NV signal information

    try:
        slm = ThorSLM(serialNumber="00429430")
        initial_phase = np.zeros(
            slm.shape
        )  # Initialize the phase array with the SLM shape
        main(slm, nv_sig, initial_phase)
    finally:
        print("Closing")
        slm.close_window()
        slm.close_device()
