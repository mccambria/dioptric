# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:06:46 2019

This routine takes the sets of data we take for relaxation measurments (prepare
in +1, readout in -1, etc) and calculates the relaxation rates, omega and
gamma. It calculates the values for each run of the data (num_runs). It will
then allow us to average the value for the relaxation rate and take a standard
deviation.

@author: Aedan
"""

import 