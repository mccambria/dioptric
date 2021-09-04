from re import search
import utils.tool_belt as tool_belt
import shelve
import os
from pathlib import PurePath

nvdata_dir = tool_belt.get_nvdata_dir()
os.chdir(nvdata_dir)
search_index_file_name = "search_index"


def gen_search_index():

    global search_index_file_name, nvdata_dir
    search_index = shelve.open(search_index_file_name)

    limit = 20
    step = 0
    str_path_nvdata_dir = str(nvdata_dir)
    for root, dirs, files in os.walk(nvdata_dir):
        for f in files:
            if step > limit:
                return
            step += 1
            str_path_root = str(PurePath(root))
            split_path_root = str_path_root.split(str_path_nvdata_dir)[1]
            # Drop the leading file separator
            index_path = split_path_root[1:]
            # print(index_path)
            # print()
            # print(f)
            # print()
            search_index[f] = index_path

    search_index.close()


def find_file(file_name):
    global search_index_file_name, nvdata_dir
    search_index = shelve.open(search_index_file_name)
    path_from_nvdata = search_index[file_name]
    search_index.close()
    file_path = nvdata_dir / path_from_nvdata / file_name
    return file_path


def gen_search_index_test(file_name, file_path):
    global search_index_file_name
    search_index = shelve.open(search_index_file_name)
    search_index[file_name] = file_path
    search_index.close()


def find_file_test(file_name):
    global search_index_file_name
    search_index = shelve.open(search_index_file_name)
    file_path = search_index[file_name]
    search_index.close()
    return file_path


if __name__ == "__main__":
    # file_name = "2021_09_03-20_36_12-hopper-search"
    file_name = "2021_03_10-all_3x120s.PNG"
    # path_from_nvdata = (
    #     "pc_hahn/branch_time-tagger-speedup/spin_echo/2021_09/{}.txt".format(
    #         file_name
    #     )
    # )
    # file_path = nvdata_dir / path_from_nvdata
    # gen_search_index_test(file_name, file_path)
    # gen_search_index()
    test = find_file(file_name)
    print(test)
