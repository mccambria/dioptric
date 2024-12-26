# -*- coding: utf-8 -*-
"""
Created on Fall, 2024
@author: Saroj Chand
"""

import os
import sys
import time
import traceback
from datetime import datetime
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from utils import data_manager as dm


def create_movie(img_arrays, output_filename="movie.gif", nv_index=0, fps=5):
    """
    Generate a movie of NV images along step indices and save it as a GIF.

    Parameters:
        img_arrays (np.ndarray): A 4D NumPy array of shape [nv_ind, step_ind, height, width].
        output_filename (str): The path to save the output movie (GIF format).
        nv_index (int): The index of the NV center to visualize.
        fps (int): Frames per second for the movie.
    """
    if img_arrays is None:
        raise ValueError("img_arrays is missing or not loaded in the data dictionary.")
    if not isinstance(img_arrays, np.ndarray):
        raise ValueError("img_arrays must be a NumPy array.")
    if img_arrays.ndim != 4:
        raise ValueError(
            f"img_arrays must have 4 dimensions (nv_ind, step_ind, height, width). "
            f"Current dimensions: {img_arrays.ndim}"
        )
    print("img_arrays shape is valid:", img_arrays.shape)
    # Validate img_arrays structure
    if img_arrays.ndim != 4:
        raise ValueError(
            "img_arrays must have the shape [nv_ind, step_ind, height, width]."
        )

    num_steps = img_arrays.shape[1]

    # Set up the figure for visualization
    fig, ax = plt.subplots()
    img_display = ax.imshow(
        img_arrays[nv_index, 0, :, :], cmap="viridis", interpolation="nearest"
    )
    ax.set_title(f"NV {nv_index} - Step 0")
    plt.colorbar(img_display, ax=ax)

    # Define the update function for animation
    def update(frame):
        img_display.set_data(img_arrays[nv_index, frame, :, :])
        ax.set_title(f"NV {nv_index} - Step {frame}")
        return (img_display,)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_steps, interval=1000 // fps, blit=True)

    # Save the animation as a GIF
    writer = PillowWriter(fps=fps)
    ani.save(output_filename, writer=writer)
    print(f"Movie successfully saved to {output_filename}")


if __name__ == "__main__":
    # file_id = 1732403187814
    file_id = 1732408444047
    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=True)
    img_arrays = data["img_arrays"]
    print("Data keys:", data.keys())
    print("Type of img_arrays:", type(data["img_arrays"]))
    # Load img_arrays if it's a string
    if isinstance(data["img_arrays"], str):
        img_arrays_file = data["img_arrays"]
        print("img_arrays file path:", img_arrays_file)

        try:
            with np.load(img_arrays_file) as npz_file:
                print("Keys in .npz file:", npz_file.keys())
                img_arrays = npz_file["arr_0"]  # Replace 'arr_0' with the correct key
                print("Loaded img_arrays shape:", img_arrays.shape)
        except Exception as e:
            raise ValueError(
                f"Failed to load img_arrays from file '{img_arrays_file}': {e}"
            )
    else:
        img_arrays = data["img_arrays"]
        create_movie(img_arrays, output_filename="nv_movie.gif", nv_index=0, fps=5)
