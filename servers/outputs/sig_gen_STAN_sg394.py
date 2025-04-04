# -*- coding: utf-8 -*-
"""
Output server for the SRS SG394 microwave signal generator.

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = sig_gen_STAN_sg394
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import logging
import socket
import time

import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from servers.outputs.interfaces.sig_gen_vector import SigGenVector
from utils import common
from utils import tool_belt as tb


class SigGenStanSg394(LabradServer, SigGenVector):
    name = "sig_gen_STAN_sg394"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_visa"]
        di_clock = config["Wiring"]["Daq"]["di_clock"]
        resource_manager = visa.ResourceManager()
        self.sig_gen = resource_manager.open_resource(device_id)
        # Set the VISA read and write termination. This is specific to the
        # instrument - you can find it in the instrument's programming manual
        self.sig_gen.read_termination = "\r\n"
        self.sig_gen.write_termination = "\r\n"
        # Set our channels for FM
        self.daq_di_pulser_clock = di_clock
        # self.daq_ao_sig_gen_mod = config[2]
        self.task = None  # Initialize state variable
        self.reset(None)
        self._set_freq(2.87)
        logging.info("Init complete")

    @setting(0)
    def uwave_on(self, c):
        """Turn on the signal. This is like opening an internal gate on
        the signal generator.
        """

        self.sig_gen.write("ENBR 1")

    @setting(1)
    def uwave_off(self, c):
        """Turn off the signal. This is like closing an internal gate on
        the signal generator.
        """

        self.sig_gen.write("ENBR 0")

    @setting(2, freq="v[]")
    def set_freq(self, c, freq):
        """Set the frequency of the signal.

        Params
            freq: float
                The frequency of the signal in GHz
        """
        self._set_freq(freq)

    def _set_freq(self, freq):
        # Determine how many decimal places we need
        precision = len(str(freq).split(".")[1])
        self.sig_gen.write("FREQ {0:.{1}f} GHZ".format(freq, precision))
        time.sleep(0.01)

    @setting(3, amp="v[]")
    def set_amp(self, c, amp):
        """Set the amplitude of the signal.

        Params
            amp: float
                The amplitude of the signal in dBm
        """

        # Determine how many decimal places we need
        precision = len(str(amp).split(".")[1])
        cmd = "AMPR {0:.{1}f} DBM".format(amp, precision)
        # logging.info(cmd)
        self.sig_gen.write(cmd)

    @setting(5)
    def mod_off(self, c):
        """Turn off the modulation."""

        self.sig_gen.write("MODL 0")
        task = self.task
        if task is not None:
            task.close()

    @setting(7)
    def load_iq(self, c):
        """
        Set up external IQ modulation
        """

        # The sg394 only supports up to 10 dBm of power output with IQ modulation
        # Let's check what the amplitude is set as, and if it's over 10 dBm,
        # we'll quit out and save a note in the labrad logging
        if float(self.sig_gen.query("AMPR?")) > 10:
            msg = (
                "IQ modulation on sg394 supports up to 10 dBm. The power was"
                " set to {} dBm and the operation was stopped.".format(
                    self.sig_gen.query("AMPR?")
                )
            )
            raise Exception(msg)
            return

        # QAM is type 7
        self.sig_gen.write("TYPE 7")
        # STYP 1 is vector modulation
        # self.sig_gen.write('STYP 1')
        # External mode is modulation function 5
        self.sig_gen.write("QFNC 5")
        # Turn on modulation
        cmd = "MODL 1"
        self.sig_gen.write(cmd)
        # logging.info(cmd)

    # @setting(7, carrier_freq="v[]", offset_I="v[]", offset_Q="v[]")
    # def load_iq(self, c, carrier_freq, offset_I, offset_Q):
    #     """
    #     Set up internal IQ modulation.

    #     Parameters:
    #         carrier_freq: float
    #             Carrier frequency in GHz.
    #         freq_I: float
    #             Frequency for the I-channel modulation (MHz).
    #         freq_Q: float
    #             Frequency for the Q-channel modulation (MHz).
    #     """
    #     # Ensure that the signal generator does not exceed 10 dBm with IQ modulation
    #     if float(self.sig_gen.query("AMPR?")) > 10:
    #         msg = (
    #             "IQ modulation on SG394 supports up to 10 dBm. The power was"
    #             " set to {} dBm and the operation was stopped.".format(
    #                 self.sig_gen.query("AMPR?")
    #             )
    #         )
    #         raise Exception(msg)
    #     # Enable IQ modulation
    #     self.sig_gen.write("MODL 1")
    #     # Set carrier frequency (GHz)
    #     self.sig_gen.write(f"FREQ {carrier_freq} GHz")
    #     # Use internal IQ modulation mode
    #     self.sig_gen.write("QFNC 7")  # INTERNAL Cosine/Sine
    #     # Apply optional I/Q offsets (-5% to +5%)
    #     self.sig_gen.write(f"OFSI {offset_I}")  # in %
    #     self.sig_gen.write(f"OFSQ {offset_Q}")  # in %
    #     # Enable RF output for IQ modulation

    @setting(8, carrier_freq="v[]", deviation="v[]")
    def load_fm(self, c, carrier_freq, deviation):
        """
        Set up frequency modulation using a nexternal analog source
        Parameters
        ----------
        deviation : float
            The deviation ofthe frequency, in MHz. Max value is 6 MHz.

        Returns
        -------
        None.

        """
        # logging.info("test")

        # FM is type 1
        self.sig_gen.write("TYPE 1")
        # STYP 1 is analog modulation
        self.sig_gen.write("STYP 0")
        # # external is 5
        # self.sig_gen.write("MFNC 5")
        # Interanl sine is 0
        self.sig_gen.write("MFNC 0")
        # # set the rate? For external this is 100 kHz
        # self.sig_gen.write("RATE 100 kHz")
        # set the rate? For external this is 100 kHz
        self.sig_gen.write("RATE 10 kHz")
        self.sig_gen.write(f"FREQ {carrier_freq} GHz")
        # set the deviation
        cmd = "FDEV {} MHz".format(deviation)
        self.sig_gen.write(cmd)
        # Turn on modulation
        cmd = "MODL 1"
        self.sig_gen.write(cmd)

    @setting(6)
    def reset(self, c):
        self.sig_gen.write("FDEV 0")
        cmd = "MODL 0"
        self.sig_gen.write(cmd)
        self.uwave_off(c)
        self.mod_off(c)


__server__ = SigGenStanSg394()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
