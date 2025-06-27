# this is the command that will be sent to the server to get a temp
import datetime
import os
import time

from utils import common

cmd = b"4A?\n"

cxn = common.labrad_connect()
opx = cxn.temp_monitor_SRS_ptc10
output_file = ""
# Os specific, edit as needed

# edit this as needed
nv_folder_path = "G:\\Enclosure_Temp"
while True:
    if (
        datetime.datetime.now().strftime("%m%Y")
        != os.path.split(os.path.split(output_file)[0])[1]
    ):
        output_file = os.path.join(
            nv_folder_path,
            datetime.datetime.now().strftime("%m%Y"),
            "temp_data",
        )
    if not os.path.isdir(
        os.path.join(nv_folder_path, datetime.datetime.now().strftime("%m%Y"))
    ):
        os.mkdir(os.path.join(nv_folder_path, datetime.datetime.now().strftime("%m%Y")))
    file = open(output_file, "a")
    temp = opx.get_temp(cmd)
    print(str(temp) + "," + datetime.datetime.now().strftime("%d:%H:%M:%S"), file=file)
    file.close()
    time.sleep(60 * 15)
