# -*- coding: utf-8 -*-
"""
Second SRS SG394 server

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = sig_gen_STAN_sg394_2
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

from servers.outputs.sig_gen_STAN_sg394 import SigGenStanSg394


class SigGenStanSg3942(SigGenStanSg394):
    name = "sig_gen_STAN_sg394_2"


__server__ = SigGenStanSg3942()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
