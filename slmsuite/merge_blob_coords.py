import numpy as np

# Path for saving and loading the data
path = "slmsuite/nv_blob_detection"

# Load coordinates from two or more files
data1 = np.load(f"{path}/nv_blob_filtered.npz")
data2 = np.load(f"{path}/nv_blob_filtered_162nvs.npz")

coords1 = data1["nv_coordinates"]
coords2 = data2["nv_coordinates"]

# Reference NV coordinate to keep as the first in the merged list
reference_nv = [113.431, 149.95]


# Define a function to calculate the Euclidean distance between two coordinates
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Start merged coordinates with the reference NV
merged_coords = [reference_nv]

# Append coordinates from the first file if they are not duplicates of the reference NV
for coord in coords2:
    if (
        euclidean_distance(coord, reference_nv) >= 3
    ):  # Remove if too close to reference NV
        merged_coords.append(coord.tolist())

# Check and append coordinates from the second file if not within 3 pixels of any in the merged list
# for coord in coords2:
#     keep_coord = True
#     for existing_coord in merged_coords:
#         if euclidean_distance(coord, existing_coord) < 3:
#             keep_coord = False
#             break
#     if keep_coord:
#         merged_coords.append(coord.tolist())

# Convert the list back to a numpy array for saving
merged_coords_array = np.array(merged_coords)

# Save the combined coordinates to a new file
np.savez(
    f"{path}/nv_blob_filtered_162nvs_ref.npz",
    nv_coordinates=merged_coords_array,
)

print(f"Number of NVs after merging: {len(merged_coords_array)}")
print("Merged NV coordinates:", merged_coords_array)
