# -*- coding: utf-8 -*-
"""
Calculators for Lakeshore X162690 temp sensor

Created on December 8th, 2022

@author: mccambria
"""

import numpy as np

def cheb(res):
    """Chebyshev function, 80.1 to 325 K"""
    
    
    ZL = 1.80004650154
    ZU = 2.41462096894
    coeffs = [177.081586, -126.684572, 22.375083, -3.178372, 0.587644, -0.111775, 0.015145, -0.001546]
    
    Z = np.log10(res)
    k = ((Z-ZL)-(ZU-Z)) / (ZU-ZL)
    
    cheb_vals = [coeffs[ind] * np.cos(ind * np.arccos(k)) for ind in range(len(coeffs))]
    return np.sum(cheb_vals)

if __name__ == "__main__":
    
    print(cheb(190.1))
    