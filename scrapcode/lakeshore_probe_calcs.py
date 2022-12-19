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

def cheb2(res):
    """Chebyshev function, 1.40 to 14.2 K"""
    
    ZL = 2.91027046513 
    ZU = 4.50588986348
    coeffs = [5.347944, -6.189287 , 2.900256, -1.177309 , 0.419822, -0.129649 , 0.031580, -0.003947 , -0.000836 , 0.002030]
    
    Z = np.log10(res)
    k = ((Z-ZL)-(ZU-Z)) / (ZU-ZL)
    
    cheb_vals = [coeffs[ind] * np.cos(ind * np.arccos(k)) for ind in range(len(coeffs))]
    return np.sum(cheb_vals)

if __name__ == "__main__":
    
    # print(cheb(69.7))
    # print(cheb(70.03))
    
    print(cheb2(1176.3))
    
    