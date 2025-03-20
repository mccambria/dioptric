# -*- coding: utf-8 -*-



from ftplib import FTP

class Indexer:
    index_filename = "index"
    index = dict()
    
    def __init__(self, ftp, update=False):
        index = BytesIO()
        if(update):
            with open(index_filename, "wb") as file:
                ftp.retrbinary(f"RETR {index_filename}", file.write)
        
        
        

    def gen_search_index(directory):
    #Gen the search index by recursively traversing the ftp server and creating a new index
        index = dict()
        ftp.cwd(directory)
        for entry in ftp.mlsd():
            if entry[1]['type'] == 'dir':
                remotepath = remotedir + "/" + entry[0]
                list_recursive(ftp, remotepath)
            else:
                index[entry[0]] = directory + "/" + name
        #todo, write and upload index


    def load_index():
        #loads index into a dictionary
        with open(index_filename) as file:
            for line in file:
                (key, val) = line.split()
                index[key] = val


    def add_to_search_index(data_full_path):
        #downloads latest index, checks if file exits, if not adds it
        return
    





    def get_data_path(data_file_name):
        if data_file_name in index:
            return index[data_file_name]
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_file_name)
   





    if __name__ == "__main__":


        gen_search_index()



    # root = nvdata_dir / "pc_hahn/branch_master/pulsed_resonance/2021_09"

    # # root = nvdata_dir / PurePath("pc_hahn", "branch_master", "pulsed_resonance", "2021_09")

    # files = [

    #     "2021_09_13-15_29_34-hopper-search.txt",

    #     "2021_09_13-15_41_02-hopper-search.txt",

    # ]

    # paths = [root / el for el in files]



    # # print(search_index_glob)

    # for el in paths:

    #     # print(el)

    #     # print(el.match(search_index_glob))

    #     add_to_search_index(el)