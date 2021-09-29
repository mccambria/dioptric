from re import search
import utils.tool_belt as tool_belt
import shelve
import os
from pathlib import PurePath
import time
import re
import sqlite3

nvdata_dir = tool_belt.get_nvdata_dir()
os.chdir(nvdata_dir)
search_index_file_name = "search_index.db"


def gen_search_index():
    
    global search_index_file_name, nvdata_dir

    # Create the table
    search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
    cursor = search_index.cursor()

    # Create table
    cursor.execute("""CREATE TABLE search_index (file_name text, path_from_nvdata text)""")

    limit = 100
    step = 0
    str_path_nvdata_dir = str(nvdata_dir)
    # return
    for root, dirs, files in os.walk(nvdata_dir):
        for f in files:
            # Only index data files in their original locations
            regex = "{}\/pc_[a-z]+\/branch_[a-z\-]+\/[a-z\_]+\/[0-9]{{4}}_[0-9]{{2}}".format(str_path_nvdata_dir)
            if not re.match(regex, root):
                continue
            if f.endswith(".txt"):
                # if step > limit:
                #     return
                # print(root)
                # print(type(root))
                # step += 1
                str_path_root = str(PurePath(root))
                split_path_root = str_path_root.split(str_path_nvdata_dir)[1]
                # Drop the leading file separator
                index_path = split_path_root[1:]
                index_file_name = f.split(".")[0]
                # print(index_path)
                # print()
                # print(f)
                # print()
                cursor.execute("INSERT INTO search_index VALUES (?, ?)", (index_file_name, index_path))

    search_index.commit()
    search_index.close()


def find_file(file_name):
    global search_index_file_name, nvdata_dir
    try:
        search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
        cursor = search_index.cursor()
        cursor.execute("SELECT * FROM search_index WHERE file_name = '{}'".format(file_name))
        res = cursor.fetchone()
        return res[1]
    except Exception as exc:
        print(
            "Failed to find file using search index. Try re-compiling the"
            " index by running gen_search_index."
        )
        return None


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
    # file_name = "2021_09_03-20_36_12-hopper-search.txt"
    # file_name = "2021_03_10-all_3x120s.PNG"
    file_name = "2021_09_03-20_36_12-hopper-search.txt"
    # path_from_nvdata = (
    #     "pc_hahn/branch_time-tagger-speedup/spin_echo/2021_09/{}.txt".format(
    #         file_name
    #     )
    # )
    # file_path = nvdata_dir / path_from_nvdata
    # gen_search_index_test(file_name, file_path)
    # gen_search_index()
    start = time.time()
    test = find_file(file_name)
    stop = time.time()
    print(stop - start)
    print(test)
