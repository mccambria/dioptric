# -*- coding: utf-8 -*-
"""

Parsing data from a txt file. This is made for a spin echo experiment that
ran into an error partway through the experiment. 

This file will go through that line in the saved iPython console and add the
data to their respective lists. It will then save that file.

This could be expanded to be more useful outside of just this specific case.



Created on Fri Jul 19 15:44:48 2019

@author: gardill
"""

import numpy


# %% File to use

file = open('C:/Users/kolkowitz/Desktop/iPython_console/test.txt', 'r')

# %% Create some lists to fill

data = numpy.empty([29, 101, 3])

ind_first = 0
ind_sec = 1
ind_run = -1

for line in file:
    
    if line[0:10] == 'Run index:':
        run_ind = line[11]
        ind_run =+ 1
        
    if line[0:22] == 'First relaxation time:':
        first_rel_time = line[23:29]
#        print('1 time {}'.format(first_rel_time))
        
    if line[0:23] == 'Second relaxation time:':
        second_rel_time = line[24:30]
#        print('2 time {}'.format(second_rel_time))
        
    if line[0:12] == 'First signal':
        frst_sig = line[15:20]
#        print('1 sig {}'.format(frst_sig))
        
    if line[0:15] == 'First Reference':
        frst_ref = line[18:23]
#        print('1 ref {}'.format(frst_ref))
        
    if line[0:13] == 'Second Signal':
        scnd_sig = line[16:21]
#        print('2 sig {}'.format(scnd_sig))
        
    if line[0:16] == 'Second Reference':
        scnd_ref = line[19:24]
#        print('2 ref {}'.format(scnd_ref))
        
        