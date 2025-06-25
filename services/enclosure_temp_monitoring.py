import time
import serial
import datetime
import os


#Edit these as needed
#On linux, you may have to give perms for this, use chmod 666
port = "/dev/ttyUSB0"
baudrate = 9600
output_file = "/home/eric/test"
#Os specific, edit as needed

#edit this as needed
nv_folder_path = "/path/to/nv"

ser = serial.Serial(port, baudrate=baudrate)
#writes command to get output of 4A -- where the temp controller is connected to
#See the ptc10 manual for more info


#We will be running this constantly
while True:
    if (datetime.datetime.now().strftime("%M%Y") != os.path.split(os.path.split(output_file)[0])[1]):
        output_file = os.path.join(nv_folder_path, datetime.datetime.now().strftime("%M%Y"), "temp_data")
        if not os.path.isdir(os.path.join(nv_folder_path, datetime.datetime.now().strftime("%M%Y"))):
            os.mkdir(os.path.join(nv_folder_path, datetime.datetime.now().strftime("%M%Y")))
    file = open(output_file, "a")
    ser.write(b'4A?\n')
    out = ''
    time.sleep(1)
    data = b''
    while not data:
        data = ser.readline()
        if len(data) > 0:
        #grabs the int value of temp
            result = float(data.split(b'\r')[0])
            print(str(result) + "," + datetime.datetime.now().strftime("%d:%H:%M:%S") + "\n", file=file)
            file.close()
    time.sleep(60)
ser.close()