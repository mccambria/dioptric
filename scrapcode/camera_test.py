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
    img_array = cam.get_img_array()
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
