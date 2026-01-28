from utils import common
import time
cxn = common.labrad_connect()
opx = cxn.filter_slider_THOR_ell9k

i = 0
while True:
    pos = i % 4
    print(f"Setting filter to position {pos}")
    opx.set_filter(pos)
    time.sleep(5)
    i += 1
