# -*- coding: utf-8 -*-
"""
Here are functions for our search index, which allows us to quickly look up
data files without specifying the file path. All you need is the file name!

Created 2021_09_10

@author: mccambria
"""

import utils.common as common
import os
from pathlib import PurePath, PurePosixPath
import re
import sqlite3

search_index_file_name = "search_index.db"
nvdata_dir = common.get_nvdata_dir()
nvdata_dir_str = str(nvdata_dir)
date_glob = "[0-9][0-9][0-9][0-9]_[0-9][0-9]"
search_index_glob = "{}/pc_*/branch_*/*/{}/*.txt".format(nvdata_dir_str, date_glob)

def process_full_path(full_path):
    """Return just what we want for writing to the database. Expects a string
    containing the entire path to the file, including nvdata, the file 
    name, the extension..."""
    
    # Make sure we have a PurePath to manipulate
    full_path = PurePath(full_path)
    
    # Get the path to the file separated from the file name itself and nvdata
    path_to_file = full_path.parent
    path_to_file_parts = path_to_file.parts
    nvdata_ind = path_to_file_parts.index("nvdata")
    index_path_parts = path_to_file_parts[nvdata_ind+1:]
    index_path = PurePath(index_path_parts[0]).joinpath(*index_path_parts[1:])
    index_path = str(index_path)
    
    # Get the file name, no extension
    index_file_name = full_path.stem
    
    return (index_file_name, index_path)
    

def gen_search_index():
    """Create the search index from scratch. Does not delete the existing 
    index so you should probably do that. Just delete the search_index files
    in nvdata. This will take several minutes.
    """
    
    # Create the table
    search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
    cursor = search_index.cursor()
    cursor.execute("""CREATE TABLE search_index (file_name text, path_from_nvdata text)""")

    for root, _, files in os.walk(nvdata_dir):
        for f in files:
            # Only index data files in their original locations
            if PurePath(f).match(search_index_glob):
                db_vals = process_full_path("{}/{}".format(root, f))
                cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)

    search_index.commit()
    search_index.close()
    
def add_to_search_index(data_full_path):
    db_vals = process_full_path(data_full_path)
    search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
    cursor = search_index.cursor()
    cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)
    search_index.commit()
    search_index.close()


def get_data_path(data_file_name):
    try:
        search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
        cursor = search_index.cursor()
        cursor.execute("SELECT * FROM search_index WHERE file_name = '{}'".format(data_file_name))
        res = cursor.fetchone()
        return res[1]
    except Exception as exc:
        print(
            "Failed to find file using search index. Try re-compiling the"
            " index by running gen_search_index."
        )
        return None


if __name__ == "__main__":
    
    # gen_search_index()
    
    root = nvdata_dir / "pc_hahn/branch_master/pulsed_resonance/2021_09"
    # root = nvdata_dir / PurePath("pc_hahn", "branch_master", "pulsed_resonance", "2021_09")
    files = [
        "2021_09_13-15_29_34-hopper-search.txt",
        "2021_09_13-15_41_02-hopper-search.txt",
    ]
    paths = [root / el for el in files]
    
    # print(search_index_glob)
    for el in paths:
        # print(el)
        # print(el.match(search_index_glob))
        add_to_search_index(el)