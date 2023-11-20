# -*- coding: utf-8 -*-
"""
Config file for the PC knuth - redirects to Rabi

Created July 20th, 2023

@author: mccambria
"""

from config.rabi import config


if __name__ == "__main__":
    print(config["Camera"])
