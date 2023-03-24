# -*- coding: utf-8 -*-
"""
Interface for arbitrary waveform generators

Created on November 29th, 2022

@author: gardill
"""

from abc import ABC, abstractmethod
# from labrad.server import LabradServer
from labrad.server import setting
import numpy
from numpy import pi

root2_on_2 = numpy.sqrt(2) / 2
# amp = 0.5  # from SRS sig gen datasheet, ( I^2 + Q^2 ) ^ (1/2) = 0.5 V for full scale input. The amp should then be 0.5 V. This relates to 1.0 Vpp from the AWG

    
class AWG(ABC):
    

    @setting(2)
    def set_i_full(self, c):
        '''
        Set the I component fully, and the Q component to 0
        '''

        amp = self.iq_comp_amp
        self.load_iq([0], amp)
        
    @setting(10, phases="*v[]")
    def load_arb_phases(self, c, phases):
        '''
        Load an arbitrary list of IQ phases
        '''

        phases_list = [0]

        for el in phases:
            phases_list.append(el)

        amp = self.iq_comp_amp
        self.load_iq(phases_list, amp)
        
    @setting(3)
    def load_knill(self, c):
        '''
        Load knill pulses for a pi pulse
        '''

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
        
    @setting(11, num_dd_reps="i")
    def load_xy4n(self, c, num_dd_reps):
        '''
        Load phases for XY4, which should be:
            [0, (0, pi/2, 0, pi/2, 0, pi/2, 0, pi/2)*N, 0]
        '''

        phases = [0] + [0, pi / 2, 0, pi / 2] * num_dd_reps + [0]
        phases = phases * 2
        phases = [0] + phases

        amp = self.iq_comp_amp
        self.load_iq(phases, amp)
        
    @setting(13, num_dd_reps="i")
    def load_xy8n(self, c, num_dd_reps):
        '''
        Load phases for XY8, which should be:
            [0, (0, pi/2, 0, pi/2, 0, pi/2, 0, pi/2)*N, 0]
        '''

        # intended phase list: [0, (0, pi/2, 0, pi/2, 0, pi/2, 0, pi/2)*N, 0]
        phases = (
            [0]
            + [0, pi / 2, 0, pi / 2, pi / 2, 0, pi / 2, 0] * num_dd_reps
            + [0]
        )
        phases = phases * 2
        phases = [0] + phases

        amp = self.iq_comp_amp
        self.load_iq(phases, amp)
        
    @setting(12, num_dd_reps="i")
    def load_cpmg(self, c, num_dd_reps):
        '''
        Load phases for CPMG, which should be:
            [0, (pi/2)*N, 0]
        '''

        # intended phase list: [0, (pi/2)*N, 0]

        phases = [0] + [pi / 2] * num_dd_reps + [0]  ###
        
        # 11/20/2022 Tried alternating phase, but for N>4, state is not coherent
        # half_num_dd_reps = int(num_dd_reps/2)
        # phases = [0] +  [pi/2, 3*pi/2]*half_num_dd_reps + [0]###
        # phases = [0] +  [pi/2, -pi/2]*half_num_dd_reps + [0]###

        phases = phases * 2
        phases = [0] + phases
        amp = self.iq_comp_amp
        self.load_iq(phases, amp)
        
    # @setting(112, num_dd_reps="i")
    # def load_cpmg_dq(self, c, num_dd_reps):
    #     '''
    #     Load phases for CPMG, which should be:
    #         [0, (pi/2)*N, 0]
    #     '''

    #     # intended phase list: [0, (pi/2)*N, 0]

    #     phases = [0] + [pi] * num_dd_reps + [0]  ###
    #     # 11/20/2022 Tried alternating phase, but for N>4, state is not coherent
    #     # half_num_dd_reps = int(num_dd_reps/2)
    #     # phases = [0] +  [pi/2, 3*pi/2]*half_num_dd_reps + [0]###
    #     # phases = [0] +  [pi/2, -pi/2]*half_num_dd_reps + [0]###

    #     phases = phases * 4
    #     amp = self.iq_comp_amp
    #     self.load_iq(phases, amp)
        
    @abstractmethod
    def load_iq(self, phases, amp):
        """
        Load IQ modulation
        """
        pass
        
    @abstractmethod
    def wave_off(self, c):
        """
        Turn off the AWG
        """
        pass

    @abstractmethod
    def reset(self, c):
        """
        Make sure the device is in a neutral state for the next experiment
        """
        pass
    