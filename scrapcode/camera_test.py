# -*- coding: utf-8 -*-
"""
Testing out Nuvu camera LabRAD server

Created on August 8th, 2023

@author: mccambria
"""

from utils import tool_belt as tb
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
import labrad


def main(cxn):
    cam = cxn.camera_NUVU_hnu512gamma
    pulse_gen = cxn.pulse_gen_SWAB_82
    # print(cam.get_detector_temp())
    # print(cam.get_size())

    seq_args = [0, 1e7, "laser_INTE_520", 0]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout-camera.py"

    cam.arm()
    pulse_gen.stream_immediate(seq_file, seq_args_string, 100)
    img_array = cam.read()
    cam.disarm()

    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array)


if __name__ == "__main__":
    kpl.init_kplotlib()

    try:
        with labrad.connect(username="", password="") as cxn:
            main(cxn)
    finally:
        tb.reset_cfm()

    plt.show(block=True)
