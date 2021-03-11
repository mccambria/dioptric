# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:42:58 2021

@author: mccambria
"""


# %% Imports


import serial


# %% Main


def main():
    
    dev = serial.Serial('COM3', 9600, serial.EIGHTBITS,
                        serial.PARITY_NONE, serial.STOPBITS_ONE)
    print(dev)
    
    try:
        pass
        # dev.read()
        # print(dev.readline().decode())
    finally:
        dev.close()


# %% Run the file


if __name__ == '__main__':
    
    main()
    