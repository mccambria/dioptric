import time

import numpy as np

from utils import positioning as pos

# Reset to origin coordinates
# time.sleep(1.0)
# print(f"Resetting XYZ coordinates to: {reset_coords}")
# pos.set_xyz(reset_coords)
reset_coords = [0.00, 0.00, 0.1]
pos.set_xyz(reset_coords, pos="pos_xyz")

pos.get_positioner_write_fn(1, pos="pos_xyz")

# def test_write_xyz_loop():
#     # Define the ranges and steps for the test loop
#     x_range = np.linspace(-3.0, 3.0, num=10)
#     y_range = np.linspace(-3.0, 3.0, num=10)
#     z_range = np.linspace(-3.0, 3.0, num=10)
#     sleep_time = 0.5
#     for x in x_range:
#         coords = [x, 0.0, 0]
#         print(f"writing x coord to:{coords}")
#         time.sleep()
#         pos.get_axis_write_fn(coords[0], coords[1], coords[2])
#         time.sleep(sleep_time)
#     print("Test loop in x is completed")
#     # Iterate over the ranges
#     for x in x_range:
#         for y in y_range:
#             for z in z_range:
#                 coords = [x, y, z]
#                 print(f"Setting XYZ coordinates to: {coords}")
#                 pos.set_xyz(coords)
#                 time.sleep(
#                     sleep_time
#                 )  # Sleep to allow the system to process the update


#     print("Test loop completed successfully.")
# reset

# if __name__ == "__main__":
#     test_write_xyz_loop()
