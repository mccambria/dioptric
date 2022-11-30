# -*- coding: utf-8 -*-
"""
Output server for the arbitrary waveform generator.

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = awg_KEYS_33622A
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


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import socket
import logging
import time
import numpy
from numpy import pi

root2_on_2 = numpy.sqrt(2) / 2
# amp = 0.5  # from SRS sig gen datasheet, ( I^2 + Q^2 ) ^ (1/2) = 0.5 V for full scale input. The amp should then be 0.5 V. This relates to 1.0 Vpp from the AWG


def iq_comps(phase, amp):
    if type(phase) is list:
        ret_vals = []
        for val in phase:
            ret_vals.append(numpy.round(amp * numpy.exp((0 + 1j) * val), 5))
        return (numpy.real(ret_vals).tolist(), numpy.imag(ret_vals).tolist())
    else:
        ret_val = numpy.round(amp * numpy.exp((0 + 1j) * phase), 5)
        return (numpy.real(ret_val), numpy.imag(ret_val))


class AwgKeys33622A(LabradServer):
    name = "awg_KEYS_33622A"
    pc_name = socket.gethostname()

    def initServer(self):
        filename = (
            "E:/Shared drives/Kolkowitz Lab"
            " Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(["", "Config", "DeviceIDs"])
        p.get("arb_wave_gen_visa_address")
        p.cd(["", "Config", "Wiring", "PulseGen"])
        p.get("do_arb_wave_trigger")
        p.cd(["", "Config", "Microwaves"])
        p.get("iq_comp_amp")
        result = await p.send()
        return result["get"]

    def on_get_config(self, config):
        address = config[0]
        self.do_arb_wave_trigger = int(config[1])
        resource_manager = visa.ResourceManager()
        self.wave_gen = resource_manager.open_resource(address)
        self.iq_comp_amp = config[2]
        self.reset(None)
        logging.info("Init complete")

    @setting(2, amp="v[]")
    def set_i_full(self, c, amp):

        self.load_iq([0], amp)

    @setting(3)
    def load_knill(self, c):

        # There's a minimum number of points, thus * 16
        # phases = [0, +pi/2, 0] * 16
        phases = [
            pi / 6,
            0,
            pi / 2,
            0,
            pi / 6,
            # pi/6+pi, 0+pi, pi/2+pi, 0+pi, pi/6+pi] * 8
            pi / 6 + pi / 2,
            0 + pi / 2,
            pi / 2 + pi / 2,
            0 + pi / 2,
            pi / 6 + pi / 2,
        ] * 8
        # phases = [0, -pi/2, 0,
        #           pi/2, 0, pi/2,
        #           3*pi/2, pi, 3*pi/2,
        #           pi, pi/2, pi] * 4

        amp = self.iq_comp_amp
        self.load_iq(phases, amp)

    @setting(10, phases="*v[]")
    def load_arb_phases(self, c, phases):

        phases_list = []

        for el in phases:
            phases_list.append(el)

        amp = self.iq_comp_amp
        self.load_iq(phases_list, amp)

    @setting(11, num_dd_reps="i")
    def load_xy4n(self, c, num_dd_reps):

        # intended phase list: [0, (0, pi/2, 0, pi/2, 0, pi/2, 0, pi/2)*N, 0]
        phases = [0] + [0, pi / 2, 0, pi / 2] * num_dd_reps + [0]
        phases = phases * 4

        amp = self.iq_comp_amp
        self.load_iq(phases, amp)

    @setting(13, num_dd_reps="i")
    def load_xy8n(self, c, num_dd_reps):

        # intended phase list: [0, (0, pi/2, 0, pi/2, 0, pi/2, 0, pi/2)*N, 0]
        phases = (
            [0]
            + [0, pi / 2, 0, pi / 2, pi / 2, 0, pi / 2, 0] * num_dd_reps
            + [0]
        )
        phases = phases * 4

        amp = self.iq_comp_amp
        self.load_iq(phases, amp)

    @setting(12, num_dd_reps="i")
    def load_cpmg(self, c, num_dd_reps):

        # intended phase list: [0, (pi/2)*N, 0]

        phases = [0] + [pi / 2] * num_dd_reps + [0]  ###
        # 11/20/2022 Tried alternating phase, but for N>4, state is not coherent
        # half_num_dd_reps = int(num_dd_reps/2)
        # phases = [0] +  [pi/2, 3*pi/2]*half_num_dd_reps + [0]###
        # phases = [0] +  [pi/2, -pi/2]*half_num_dd_reps + [0]###

        phases = phases * 4
        amp = self.iq_comp_amp
        self.load_iq(phases, amp)

    # @setting(14)
    # def load_fsk_test(self, c):

    #     # phases = [pi/2]
    #     phases = [0]

    #     phases = phases*4
    #     amp = self.iq_comp_amp
    #     self.load_iq(phases, amp)

    def load_iq(self, phases, amp):
        """
        Load IQ modulation
        """

        self.wave_gen.write("TRIG1:SOUR EXT")
        self.wave_gen.write("TRIG2:SOUR EXT")
        self.wave_gen.write("TRIG1:SLOP POS")
        self.wave_gen.write("TRIG2:SLOP POS")

        for chan in [1, 2]:
            source_name = "SOUR{}:".format(chan)
            self.wave_gen.write("{}FUNC:ARB:FILT OFF".format(source_name))
            self.wave_gen.write("{}FUNC:ARB:ADV TRIG".format(source_name))
            self.wave_gen.write("{}FUNC:ARB:PTP 2".format(source_name))

        # There's a minimum length of points you must send, so let's just
        # repeat until it's long enough
        while len(phases) < 32:
            phases *= 2

        phase_comps = iq_comps(phases, amp)
        # logging.info(phase_comps)

        # Shift the last element to first to account for first pulse in seq
        # Then convert to string and trim the brackets
        # for the I channel
        comps = phase_comps[0]
        last_el = comps.pop()
        comps.insert(0, last_el)
        seq = str(comps)[1:-1]
        self.wave_gen.write("SOUR1:DATA:ARB iqSwitch1, {}".format(seq))

        # for the Q channel
        comps = phase_comps[1]
        last_el = comps.pop()
        comps.insert(0, last_el)
        seq = str(comps)[1:-1]
        self.wave_gen.write("SOUR2:DATA:ARB iqSwitch2, {}".format(seq))

        for chan in [1, 2]:
            source_name = "SOUR{}:".format(chan)
            self.wave_gen.write(
                "{}FUNC:ARB iqSwitch{}".format(source_name, chan)
            )
            self.wave_gen.write("{}FUNC ARB".format(source_name))

        self.wave_gen.write("OUTP1 ON")
        self.wave_gen.write("OUTP2 ON")

        # When you load a sequence like this, it doesn't move to the first
        # point of the sequence until it gets a trigger. Supposedly just
        # 'TRIG[1:2]' forces a trigger event, but I can't get it to work.
        # So let's just set the pulse streamer to constant for a second to
        # fake a trigger...
        time.sleep(0.1)  # Make sure everything is propagated
        ret_val = ensureDeferred(self.force_trigger())
        ret_val.addCallback(self.on_force_trigger)

    async def force_trigger(self):
        self.client.pulse_streamer.constant([self.do_arb_wave_trigger])
        # time.sleep(0.1)
        # self.client.pulse_streamer.constant()
        # time.sleep(0.1)
        # self.client.pulse_streamer.constant([self.do_arb_wave_trigger])

    def on_force_trigger(self):
        # Some delays included here to be sure everything actually happens
        time.sleep(0.1)
        self.client.pulse_streamer.constant()
        time.sleep(0.1)

    @setting(4)
    def test_sin(self, c):
        for chan in [1, 2]:
            source_name = "SOUR{}:".format(chan)
            self.wave_gen.write("{}FUNC SIN".format(source_name))
            self.wave_gen.write("{}FREQ 10000".format(source_name))
            self.wave_gen.write("{}VOLT:HIGH +0.5".format(source_name))
            self.wave_gen.write("{}VOLT:LOW -0.5".format(source_name))
        self.wave_gen.write("OUTP1 ON")
        self.wave_gen.write("SOUR2:PHAS 0")
        self.wave_gen.write("OUTP2 ON")

    @setting(5)
    def wave_off(self, c):
        self.wave_gen.write("OUTP1 OFF")
        self.wave_gen.write("OUTP2 OFF")

    @setting(6)
    def reset(self, c):
        self.wave_off(c)
        self.wave_gen.write("SOUR1:DATA:VOL:CLE")
        self.wave_gen.write("SOUR2:DATA:VOL:CLE")
        self.wave_gen.write("OUTP1:LOAD 50")
        self.wave_gen.write("OUTP2:LOAD 50")


__server__ = AwgKeys33622A()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)

    phases = [pi / 4, -pi / 4] * 16
    phase_comps = iq_comps(phases)
    seq1 = str(phase_comps[1])[1:-1]  # Convert to string and trim the brackets
    seq = "0.5, -0.5, " * 16
    seq2 = seq[:-2]

    print(seq1)
    print(seq2)

    print(seq1 == seq2)
