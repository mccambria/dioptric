import numpy as np

# Path for saving and loading the data
path = "slmsuite/nv_blob_detection"

# Load coordinates and spot sizes from the first file with allow_pickle=True
data1 = np.load(f"{path}/nv_blob_filtered_multiple_1.npz", allow_pickle=True)
print(data1.keys())

coords1 = data1["nv_coordinates"]
spot_sizes1 = data1["spot_sizes"]

# Reference NV coordinate to keep as the first in the merged list
reference_nv = [121.354, 159.075]


# Define a function to calculate the Euclidean distance between two coordinates
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Initialize merged coordinates and spot sizes with the reference NV
merged_coords = [reference_nv]
merged_spot_sizes = [
    spot_sizes1[
        np.argmin([euclidean_distance(coord, reference_nv) for coord in coords1])
    ]
]

# Append coordinates and spot sizes if they are not duplicates
for i, coord in enumerate(coords1):
    if euclidean_distance(coord, reference_nv) >= 3:  # Exclude close duplicates
        merged_coords.append(coord.tolist())
        merged_spot_sizes.append(spot_sizes1[i])

# Convert lists to numpy arrays
merged_coords_array = np.array(merged_coords)
merged_spot_sizes_array = np.array(merged_spot_sizes)

# Save the merged coordinates and spot sizes to a new file
np.savez(
    f"{path}/nv_blob_filtered_multiple_nv302.npz",
    nv_coordinates=merged_coords_array,
    spot_weights=merged_spot_sizes_array,
)

# Print results
print(f"Number of NVs after merging: {len(merged_coords_array)}")
print("Merged NV coordinates:", merged_coords_array)
print("Merged Spot Sizes:", merged_spot_sizes_array)
