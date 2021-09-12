# -*- coding: utf-8 -*-
"""
Here are functions for our search index, which allows us to quickly look up
data files without specifying the file path. All you need is the file name!

Created 2021_09_10

@author: mccambria
"""

import utils.common as common
import os
from pathlib import PurePath
import re
import sqlite3

search_index_file_name = "search_index.db"
nvdata_dir = common.get_nvdata_dir()
nvdata_dir_str = common.get_nvdata_dir_str()
search_index_regex = "{}\/pc_[a-z]+\/branch_[a-z\-]+\/[a-z\_]+\/[0-9]{{4}}_[0-9]{{2}}".format(nvdata_dir_str.replace("/", "\\/"))

def process_full_path(full_path):
    """Return just what we want for writing to the database. Expects a string
    containing the entire path to the file, including nvdata, the file 
    name, the extension..."""
    
    # Get the path to the file separated from the file name itself
    full_path_split = full_path.split("/")
    path_to_file = "/".join(full_path_split[0:-1])
    file_name = full_path_split[-1]
    str_path_root = str(PurePath(path_to_file))
    
    # Ditch nvdata
    split_path_root = str_path_root.split(nvdata_dir_str)[1]
    index_path = split_path_root[1:]
    
    # Ditch the extension
    index_file_name = file_name.split(".")[0]
    
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
            if re.match(search_index_regex, root) and f.endswith(".txt"):
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
    gen_search_index()
