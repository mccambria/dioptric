# -*- coding: utf-8 -*-



from ftplib import FTP
from io import BytesIO

class Indexer:
    index_filename = "index"
    index = dict()
    ftp = None
    def __init__(self, ftpclient, update=True):
        index = BytesIO()
        ftp = ftpclient
        if(update):
            download_index()
        
        
        
    def download_index():
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
                index[entry[0]] = directory + "/" + entry[0]
        save_index_to_file()
        upload_index()




    def load_index():
        #loads index into a dictionary
        with open(index_filename) as file:
            for line in file:
                (key, val) = line.split()
                index[key] = val


    def add_to_search_index(parent_path, file_name):
        #downloads latest index, checks if file exits, if not adds it
        download_index()
        load_index()
        if file_name not in index:
            index[file_name] = parent_path + "/" + file_name
        append_to_index(file_name, index[file_name])

    def append_to_index(key, val):
        file = open(index_filename, "a")
        f.write(key + " " + value)
        f.close()


    def save_index_to_file():
        file = open(index_filename, "w")
        for key, value in index.items():
            f.write(key + " " + value)
        f.close()


    def upload_index():
        file = open(index_filename, 'rb')
        ftp.storbinary(f'Stor {index_filename}', file)
        file.close()


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