# -*- coding: utf-8 -*-
"""
Output server for Lakeshore 218 Temperature Monitor.

Created on May 18th, 2022

@author: mccambria

### BEGIN NODE INFO
[info]
name = temp_monitor_lakeshore218
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 
### END NODE INFO
"""


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import logging
import socket
import pyvisa as visa
import time
import numpy
import math
import serial


class TempMonitorLakeshore218(LabradServer):
    name = "temp_monitor_lakeshore218"
    pc_name = socket.gethostname()
    term = "\r\n"

    def initServer(self):
        filename = (
            "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)
        
    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(["", "Config", "DeviceIDs"])
        p.get("{}_address".format(self.name))
        result = await p.send()
        return result["get"]

    def on_get_config(self, config):
        # Get the slider
        try:
            self.monitor = serial.Serial(config, 9600, serial.SEVENBITS, 
                            serial.PARITY_ODD, serial.STOPBITS_ONE, timeout=2)
        except Exception as e:
            logging.debug(e)
            del self.monitor
        time.sleep(0.1)
        self.monitor.flush()
        time.sleep(0.1)
        logging.info("Init complete")
        
        
    def write(self, cmd):
        self.monitor.write("{}{}".format(cmd, self.term).encode())
        
        
    def read(self):
        ret_val = self.monitor.readline()
        return ret_val
    
        
    @setting(6)
    def reset_cfm_opt_out(self, c):
        """This setting is just a flag for the client. If you include this 
        setting on a server, then the server won't be reset along with the 
        rest of the instruments when we call tool_belt.reset_cfm.
        """
        pass


__server__ = TempMonitorLakeshore218()

if __name__ == "__main__":
    # from labrad import util

    # util.runServer(__server__)
    
    term = "\r\n"
    
    ###################
    
    # # Set up a user curve for a calibrated temp sensor
    # sensor_num = 1
    # name = "cernox"
    # sensor_serial = "X162689"
    # curve_format = 4  # 2: V/K, 3: Ohm / K, 4: log(Ohm) / K
    # limit = 325
    # coeff = 1  # 1: ntc, 2: ptc
    # curve_res_str = ["3.81561351246670E+04", "3.03990411462499E+04", "2.48075036914347E+04", "1.78898396874006E+04", "1.35989377487676E+04", "1.09240278904390E+04", "9.04536734433402E+03", "7.67767356239885E+03", "6.66472383328352E+03", "5.88536254716986E+03", "5.28112195205923E+03", "4.76116479244462E+03", "4.35492262010328E+03", "4.00595650487687E+03", "3.71673545540780E+03", "3.46279578959874E+03", "3.23427281946515E+03", "2.90084005754182E+03", "2.61453496580513E+03", "2.33542405533906E+03", "2.04166698767839E+03", "1.79747688120349E+03", "1.57432247149231E+03", "1.40860993255677E+03", "1.28150517310290E+03", "1.18008672054988E+03", "1.09787365338608E+03", "1.02921221364105E+03", "9.71359087107854E+02", "9.21631838080876E+02", "8.77975067382375E+02", "8.39426693802750E+02", "8.04742336713244E+02", "7.73482836216998E+02", "7.44877313037891E+02", "7.16054668851399E+02", "6.77617214311519E+02", "6.43460509323363E+02", "6.12825856907732E+02", "5.84956166633408E+02", "5.59584677890930E+02", "5.33760301191417E+02", "5.06785099250299E+02", "4.73073255086253E+02", "4.43756209217353E+02", "4.18014006330343E+02", "3.95239307906806E+02", "3.74852762470300E+02", "3.62440446504042E+02", "3.34811670828641E+02", "3.11176552760606E+02", "2.90713826769879E+02", "2.72768756354097E+02", "2.56970544452674E+02", "2.42882737421126E+02", "2.30242117438531E+02", "2.18861937127253E+02", "2.08566002815496E+02", "1.99141182845699E+02", "1.82632276167560E+02", "1.68613478602219E+02", "1.56526774898855E+02", "1.46031377737805E+02", "1.36826699205152E+02", "1.28696790758208E+02", "1.21459987288770E+02", "1.14991986529460E+02", "1.09161286524191E+02", "1.03910047844886E+02", "9.91437228771382E+01", "9.48127382338156E+01", "9.08443326724668E+01", "8.72090044361032E+01", "8.38901304427628E+01", "8.07981749209235E+01", "7.79612537047131E+01", "7.53260922219343E+01", "7.28868882146118E+01", "7.06187722979619E+01", "6.85033781101720E+01", "6.74970398110578E+01", "6.65270748859460E+01", "6.54023947807793E+01", "6.46824532976803E+01"]
    # curve_temps_str = ["1.20049786574564E+00", "1.30023112622114E+00", "1.40330319656789E+00", "1.59970024565167E+00", "1.80241715845650E+00", "1.99885299091163E+00", "2.19928629209236E+00", "2.40168641211146E+00", "2.60157602711211E+00", "2.80051633416471E+00", "2.99444661364056E+00", "3.20084630328525E+00", "3.39750958577106E+00", "3.59986643203534E+00", "3.79811887282963E+00", "4.00149053517466E+00", "4.21422041681745E+00", "4.59381694761858E+00", "5.00412713961944E+00", "5.51678434730521E+00", "6.23826962607435E+00", "7.06047839210696E+00", "8.09245658113728E+00", "9.13111090036954E+00", "1.01655210295737E+01", "1.11989926111450E+01", "1.22194492754145E+01", "1.32329494112086E+01", "1.42308787850054E+01", "1.52151945436925E+01", "1.61961110533837E+01", "1.71665474163577E+01", "1.81372745134835E+01", "1.91034157310393E+01", "2.00708893651524E+01", "2.11379497856368E+01", "2.27282597243524E+01", "2.43219558465760E+01", "2.59219726669750E+01", "2.75395458103582E+01", "2.91650705930756E+01", "3.09904767175680E+01", "3.31092899797281E+01", "3.61129266313024E+01", "3.91210422865412E+01", "4.21266001363853E+01", "4.51247238172927E+01", "4.81241791425300E+01", "5.01195516182526E+01", "5.51107653259187E+01", "6.00999405397520E+01", "6.50885039443929E+01", "7.00840471754715E+01", "7.50751785385947E+01", "8.00662707081731E+01", "8.50696384532997E+01", "9.00636926750425E+01", "9.50601245977394E+01", "1.00056813174383E+02", "1.10047245390549E+02", "1.20048242625626E+02", "1.30049224696408E+02", "1.40047382972819E+02", "1.50044846611906E+02", "1.60039930543893E+02", "1.70030729224588E+02", "1.80025619479608E+02", "1.90028877056255E+02", "2.00028503995239E+02", "2.10028612192541E+02", "2.20017835226514E+02", "2.30024676295733E+02", "2.40027060663824E+02", "2.50016658742665E+02", "2.60033509526586E+02", "2.70029120825437E+02", "2.80039283319594E+02", "2.90032493311733E+02", "3.00044468282123E+02", "3.10025852862835E+02", "3.15032786333869E+02", "3.20036469359610E+02", "3.26024248244956E+02", "3.30034008452599E+02"]
    
    # # Get 6 total digits, no rounding (assumes a decimal in there somewhere)
    # round_to_6_log = lambda val: str(math.log(float(val), 10))[0:7]
    # round_to_6 = lambda val: str(float(val))[0:7]
    # curve_units = [round_to_6_log(val) for val in curve_res_str]
    # curve_temps = [round_to_6(val) for val in curve_temps_str]
    # num_points = len(curve_units)
    # if num_points != len(curve_temps):
    #     print("curve_units and curve_temps are not the same length! You really goofed up now.")
    
    # curve_num = 20 + sensor_num
    # with serial.Serial("COM1", 9600, serial.SEVENBITS, serial.PARITY_ODD, serial.STOPBITS_ONE, timeout=2) as monitor:
    #     cmd = "CRVHDR, {}, {}, {}, {}, {}, {}".format(curve_num, name, 
    #                                               sensor_serial, curve_format,
    #                                               limit, coeff)
    #     monitor.write("{}{}".format(cmd, term).encode())
    #     for ind in range(num_points):
    #     # for ind in [83]:
    #         time.sleep(0.25)
    #         unit = curve_units[ind]
    #         temp = curve_temps[ind]
    #         cmd = "CRVPT {}, {}, {}, {}".format(curve_num, ind+1, unit, temp)
    #         # cmd = "CRVPT? {}, {}".format(curve_num, ind+1)
    #         # print(cmd)
    #         monitor.write("{}{}".format(cmd, term).encode())
    #         # ret_val = monitor.readline()
    #         # print(ret_val)
    # time.sleep(0.25)
            
    ###################
    
    with serial.Serial("COM1", 9600, serial.SEVENBITS, serial.PARITY_ODD, serial.STOPBITS_ONE, timeout=2) as monitor:
    #         # cmd = "INCRV {},{}".format(sensor_num, curve_num)
        # cmd = "INCRV 1, 21"
    # #         print(cmd)
        # monitor.write("{}{}".format(cmd, term).encode())
    #         # ret_val = monitor.readline()
    #         # print(ret_val)
    #         # cmd = "INPUT 1, 1"
    #         # monitor.write("{}{}".format(cmd, term).encode())
            
    # time.sleep(0.25)
    # with serial.Serial("COM1", 9600, serial.SEVENBITS, serial.PARITY_ODD, serial.STOPBITS_ONE, timeout=2) as monitor:
    #     # cmd = "CRVHDR? 21"
    #     # cmd = "CRVPT? 21, 84"
        cmd = "KRDG? 1"
        # cmd = "SRDG? 1"
    #     # cmd = "INPUT? 1"
        # cmd = "INCRV? 1"
        monitor.write("{}{}".format(cmd, term).encode())
        ret_val = monitor.readline()
        print(ret_val)
    