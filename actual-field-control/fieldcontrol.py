#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:36 2024

@author: sean
"""
import numpy as np
import RsInstrument as Rs
from datetime import datetime

def calculate(field):
    """
    coordinate conventions (cartesian)
    x: out of the stage (controlled by PCB)
    y: "side to side" (controlled by small coils)
    z: out of the table "up and down" (controlled by big coils)
    
    first task: define what the measured field is given a 1 A current in each direction
    in an ideal world:
    x = [20,0,0]
    y = [0,20,0]
    z = [0,0,20]
    but this is not an ideal world. The diamond will not be centered, the coils and PCB produce stray fields, etc.
    
    second task: 89 and 54
    the system should behave linearly i.e. the field produced is a linear function of the input current
    we can thus use the machinery of linear algebra
    
    we can line up the vectors x,y,z as defined and shove them into a matrix 
    this matrix converts the input current to the magnetic field
    B = MI
    where M = [x y z]
    
    third task: we want to know the current we need given a desired field so the output is the matrix inverse
    I = M^-1 B
    """
    x = [19,3,2]
    y = [1,19,2]
    z = [0,1,19]
    
    M = [x, y, z] 
    M = np.transpose(M)
    
    M = np.linalg.inv(M)
    current = M.dot(field)
    
    return current


def initialize(instrumentIP=None):
    Rs.RsInstrument.assert_minimum_version('1.50.0')
    
    if instrumentIP == None:
        instr = Rs.RsInstrument('TCPIP::192.168.56.101::hislip0', True, False, "Simulate=True")
        print("the power supply " + instr.query_str('*IDN?') + " was connected at " + str(datetime.now()))
        return instr
    else:
        try:
            instr = Rs.RsInstrument(instrumentIP, True, False, "Simulate=False")
            print("the power supply " + instr.query_str('*IDN?') + " was connected at " + str(datetime.now()))
            return instr
        except:
            print("device "+ instrumentIP + " not found")
    

def xCurrent(instr,I=None):
    instr.write_str("INST OUT1")
    if I != None:
        instr.write_str("CURR "+str(I))
    else:
        return instr.query("CURR?")


def yCurrent(instr,I=None):
    instr.write_str("INST OUT2")
    if I != None:
        instr.write_str("CURR "+str(I))
    else:
        return instr.query("CURR?")

def zCurrent(instr,I=None):
    instr.write_str("INST OUT3")
    if I != None:
        instr.write_str("CURR "+str(I))
    else:
        return instr.query("CURR?")

def allCurrent(instr,I=None):
    if type(I) != type(None):
        xCurrent(instr,I[0])
        yCurrent(instr,I[1])
        zCurrent(instr,I[2])
    else:
        return [xCurrent(instr),yCurrent(instr),zCurrent(instr)]