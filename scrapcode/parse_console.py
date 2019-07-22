# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


from pathlib import Path
import numpy


# %% Constants


# %% Functions


# %% Main


def main(file_name):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
    data_dir = Path('C:/Users/kolkowitz/Desktop/iPython_console/')
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext
    with open(file_path) as file:
        console = file.read()
        
    # Build taus
    taus = []
    first_run = console.split('Run index: ')[1]
    exps = first_run.split('First relaxation time: ')
    exps = exps[1:]  # ignore the run header
    for exp in exps:
        first_time = exp.split('\n\n')[0]
        taus.append(first_time)
        second_time = exp.split('Second relaxation time: ')[1].split('\n\n')[0]
        if second_time != first_time:
            taus.append(second_time)
    taus = numpy.array(taus, dtype=numpy.int32)
    taus = numpy.sort(taus)
#    print(taus)
#    return
    
        
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    

# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    file_name = '2019-07-18_ipython'
    
    # Run the script
    main(file_name)
