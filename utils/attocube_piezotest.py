
import threading
import time

import nidaqmx
import numpy as np
from utils import common


cxn = common.labrad_connect()
opx = cxn.pos_xy_THOR_gvs212
freq_x = 4  # Hz
freq_y = 6
freq_z = 6
rate = 100  # update rate (Hz)


# === Move X, then y, then z ===
def piezo_move(stop_event):
    x_complete = False
    t = 0
    dt = 1 / rate
    
    try:
        while not stop_event.is_set():
            if(not x_complete):
                if (t >= 1/freq_x):
                    t = 0
                    x_complete = True
                    x = 0
                x += 1 * (t <= 1/(2*freq_x))+ (-1) * (t > 1/(2*freq_x))
            if(x_complete):
                if (t >= 1/freq_y):
                    t = 0
                    x_complete = False
                    y = 0
                y += 1 * (t <= 1/(2*freq_y))+ (-1) * (t > 1/(2*freq_y))
            #prevents negative steps, which may be dangerous
            if(x < 0):
                x = 0
            if(y < 0):
                y = 0
            opx.write_xy(x, y)
            time.sleep(dt)
            t += dt
    finally:
        print("Stopping scan and parking piezo at start.")
        opx.write_xy(0, 0)  #set piezo back to start


# Start piezo pattern
stop_event = threading.Event()
thread = threading.Thread(target=piezo_move, args=(stop_event,))
thread.start()

# Let it run for 10 seconds (or remove this to run indefinitely)
time.sleep(6)

# Stop
stop_event.set()
thread.join()
