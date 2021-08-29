import os
from pathlib import Path
import time
import utils.tool_belt as tool_belt
import glob

file_name = "2021_08_28-15_35_35-hopper-search"
# path = Path("/home/mccambria/E/nvdata/pc_hahn")
path = tool_belt.get_nvdata_dir()

start = time.time()

# with os.scandir(path) as it:
#     for entry in it:
#         if entry.is_dir():
#             print(entry.name)

# for dir_name, dirs, files in os.walk(path):
#     pass
# print(dir_name)
# hello = "test" in files

# year_month = file_name[0:7]
# glob_str = "{}/**/{}/{}.txt".format(str(path), year_month, file_name)
# for el in glob.glob(glob_str, recursive=True):
#     file_path = el
#     break

year_month = file_name[0:7]
glob_str = "{}/*/*/*/{}/{}.txt".format(str(path), year_month, file_name)
for el in glob.glob(glob_str):
    file_path = el
    break

stop = time.time()

print(file_path)

print(stop - start)
