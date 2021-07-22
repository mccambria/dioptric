# -*- coding: utf-8 -*-
"""
Output server for a third Thorlabs ELL9K filter slider. 
Inherits from FilterSliderEll9k

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = filter_slider_ell9k_3
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


from servers.outputs.filter_slider_ell9k import FilterSliderEll9k


class FilterSliderEll9k3(FilterSliderEll9k):
    name = 'filter_slider_ell9k_3'


__server__ = FilterSliderEll9k3()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
