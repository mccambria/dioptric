import numpy as np

nv_ind_include = []
# fmt: off
nv_ind_69MHz= [3, 7, 8, 13, 22, 25, 30, 38, 42, 50, 51, 58, 61, 76, 77, 79, 84, 88, 92, 96, 101, 105, 107, 109, 119, 128, 132, 133, 134, 139, 140, 141, 143, 145]
nv_ind_178MHz = [0, 4, 6, 14, 19, 20, 26, 31, 33, 36, 39, 43, 52, 59, 62, 63, 64, 65, 74, 75, 78, 83, 86, 90, 91, 95, 99, 110, 112, 113, 121, 126, 127, 136, 146]
# fmt: on
nv_ind_include = nv_ind_69MHz + nv_ind_178MHz
print(nv_ind_include)
nv_ind_include = sorted(nv_ind_include)

print(f"sorted nv indices = {nv_ind_include}")
