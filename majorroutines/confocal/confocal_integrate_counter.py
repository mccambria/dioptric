# Author: Eric Gediman
# Integrates total counts for a nv with provided center coordinates

import utils.data_manager as dm




if __name__ == "__main__":
    file_name = "2025_12_01-17_02_13-(Rubin)"
    data = dm.get_raw_data(file_name)
    print(list(data.keys()))