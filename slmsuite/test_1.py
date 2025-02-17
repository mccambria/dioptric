import numpy as np

nv_ind_include = []
# fmt: off
nv_ind_41MHz= [1, 5, 17, 24, 29, 35, 44, 46, 48, 49, 53, 55, 57, 58, 66, 68, 70, 72, 73, 80, 82, 93, 94, 98, 102, 103, 111, 116, 122, 124, 129, 130, 131, 138, 142]
nv_ind_178MHz = [0, 6, 10, 19, 20, 26, 36, 43, 52, 62, 64, 65, 74, 75, 78, 83, 90, 91, 95, 99, 110, 112, 113, 126, 136, 146]
# fmt: on
nv_ind_include = nv_ind_41MHz + nv_ind_178MHz
print(nv_ind_include)
nv_ind_include = sorted(nv_ind_include)

print(f"sorted nv indices = {nv_ind_include}")
