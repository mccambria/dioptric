# -*- coding: utf-8 -*-
"""

A first test to control the Cobalt lasers by python
Created on Thu Jan 14 15:41:17 2021

@author: agardill
"""
"""
* hello_laser.py
* 
*   THE PRESENT SAMPLE CODE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
*      * WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
*        TIME.
*
* The following code is released under the MIT license.
*
* Please refer to the included license document for further information.
"""
import sys
import serial
from serial import SerialException
from serial.tools import list_ports

def enumerate_ports( devices_found ):
    devices_found.extend( list_ports.comports() )
    i = 0
    for device in devices_found:
        print( "{} : ({}, {}, {})".format(i, device.device, device.description, device.hwid ) )
        i = i + 1

class InvalidPortException(Exception):
    def __str__(self):
        return "Port ID out of range"

baud = 112500
devices_found = list()

try:
    print( "Please select serial port!" )
    
    enumerate_ports( devices_found )

    port_id = int( input() )

    # Check that the user selected a valid port
    if ( port_id  >=  len( devices_found ) ):
        raise InvalidPortException()

except Exception as e:
    print(e)

# Connect to the slected port port, baudrate, timeout in milliseconds
my_serial = serial.Serial( devices_found[port_id].device, baud, timeout=1)

try:
    #Check if we managed to open the port (optional)
    print( "Is the serial port open?", end='' )
    if ( my_serial.is_open ): 
        print(" Yes.")
    else:
        print( " No." )

    # Ask the laser for its serial number, ( note the required ending \r\n )
    # Look in manual for the commands and response formatting of your laser!
    command = "gsn?"
    termination = "\r\n"
    my_serial.write( (command + termination).encode('ascii') ) # encode('ascii') converts python string to a binary ascii representation

    result = my_serial.readline().decode('ascii')

    print( "Serial number was: {}\n".format( result ))

except Exception as e:
    print(e)
    if my_serial.is_open:
        my_serial.close()
