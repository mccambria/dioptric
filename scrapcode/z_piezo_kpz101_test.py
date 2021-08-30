# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:53:06 2021

@author: kolkowitz
"""

# import labrad

# with labrad.connect() as cxn:
    
#     cxn.z_piezo_kpz101.write_z(5.0)

import ctypes

dll_path = 'C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.KCube.Piezo.dll'
piezo_lib = ctypes.windll.LoadLibrary(dll_path)
piezo_lib.TLI_BuildDeviceList()

# blank = ctypes.c_char_p()
# piezo_lib.TLI_GetDeviceListExt(ctypes.byref(blank), 100000)
# print(blank.value)

# for ind in range(int(100E6)):
#     res = piezo_lib.PCC_Open(str(ind))
#     if res != 2:
#         print(res)
#         print(ind)

serial = b"29502179"
# serial = '4122818075507898674'
serial = ctypes.c_char_p(serial)
res = piezo_lib.PCC_Open(serial)
print(res)
res = piezo_lib.PCC_StartPolling(serial, 10)
print(res)

