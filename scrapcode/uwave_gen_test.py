# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import visa


# %% Constants


# %% Functions


# %% Main


def main():
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
    resource_manager = visa.ResourceManager()
    sig_gen_address = 'TCPIP0::128.104.160.112::5025::SOCKET'
    sig_gen = resource_manager.open_resource(sig_gen_address)
    sig_gen.read_termination = '\r\n'
    sig_gen.write_termination = '\r\n'
    freq = sig_gen.query('MODL?')
    print(freq)
#    sig_gen.write('*IDN?')
#    idn = sig_gen.read_raw()
#    idn = sig_gen.query('*IDN?')
#    print(idn)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here

    # Run the script
    main()
