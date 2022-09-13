# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:19:47 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import eval_chebyt
from scipy.special import eval_chebyu
from numpy import pi

# uN = 0.762e-3 #MHz/G
# uN_13C = 0.7*uN
uN_13C = 2*pi*1.07e-3 #MH/G

Bz= 94.6 #G
f_La = 0.098 #MHz

dd_model_coeff_dict = tool_belt.get_dd_model_coeff_dict()

# def S_13C(t, N, Ax, Az):
#     t = 2*t
#     B_vec = numpy.array([0,0,Bz])
#     A_vec = numpy.array([Ax, 0, Az])
    
#     w0 = numpy.linalg.norm(uN_13C*B_vec)
#     w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
#     n0 = uN_13C*B_vec/w0
#     n1 = (uN_13C*B_vec + A_vec)/w1
    
    
#     term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
#     term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
#     phi=  numpy.arccos(term_c - numpy.dot(n0, n1) * term_s) # maybe try to avoid doing this ???


#     cross_n = numpy.cross(n0, n1)
#     term_ss = numpy.sin(w0*t*2*pi/2)**2 * numpy.sin(w1*t*2*pi/2)**2
#     term_phi = numpy.sin(N*phi/2)**2 / numpy.cos(phi / 2)**2
#     return (1 - numpy.linalg.norm(cross_n)**2 * term_ss * term_phi)

def S_13C(t, N, Ax, Az):
    t = 2*t
    B_vec = numpy.array([0,0,Bz])
    A_vec = numpy.array([Ax, 0, Az])
    
    w0 = numpy.linalg.norm(uN_13C*B_vec)
    w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
    n0 = uN_13C*B_vec/w0
    n1 = (uN_13C*B_vec + A_vec)/w1
    
    p0 = w0 * t / 2
    p1 = w1 * t / 2
    
    
    term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
    term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
    cosphi=  term_c - numpy.dot(n0, n1) * term_s


    dot_n = numpy.dot(n0, n1)
    print(dot_n)
    # print(n1)
    # term_ss = numpy.sin(w0*t*2*pi/2)**2 * numpy.sin(w1*t*2*pi/2)**2
    # term_phi = numpy.sin(N*phi/2)**2 / numpy.cos(phi / 2)**2
    k = int((N/2)-1)
    return 1 - 4*(1 - dot_n**2) * numpy.sin(p0)**2/2 * numpy.sin(p1)**2/2 *(1-cosphi) * eval_chebyu(k, cosphi)**2

def vect_mult(vector, numbers):
    '''
    mutiply a vector (numpy array) and an array of numbers
    '''
    
    return vector * numbers[:, None]

# def S_13C(t, N, Ax, Az):
#     t = 2*t
#     B_vec = numpy.array([0,0,Bz])
#     A_vec = numpy.array([Ax, 0, Az])
    
#     w0 = numpy.linalg.norm(uN_13C*B_vec)
#     w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
#     n0 = numpy.array(uN_13C*B_vec/w0)
#     n1 = numpy.array((uN_13C*B_vec + A_vec)/w1)
#     # print(n1*numpy.dot(n1,n0)-n0)
#     p0 = w0 * t *2*pi / 2
#     p1 = w1 * t *2*pi / 2
    
#     term_c = numpy.cos(p0/2) * numpy.cos(p1/2)
#     term_s = numpy.sin(p0/2) * numpy.sin(p1/2)
#     cosphi=  term_c - numpy.dot(n0, n1) * term_s
    
#     term1 =vect_mult(n0, numpy.sin(p0)*numpy.cos(p1))
#     term2= vect_mult(n1, numpy.sin(p1)*numpy.cos(p0)) 
#     term3n = vect_mult((n1*numpy.dot(n1,n0)-n0), 2*numpy.sin(p1/2)**2*numpy.sin(p0) )
#     term3m = vect_mult((n0*numpy.dot(n1,n0)-n1), 2*numpy.sin(p0/2)**2*numpy.sin(p1) )
    
#     vecn = -term1-term2+term3n
#     print(vecn)
#     vecm =  term1+term2-term3m
#     print(vecm)
    
#     dot_list = []
#     for r in range(len(vecn)):
#         v1 = vecn[r]
#         v2 = vecm[r]
#         dot = numpy.dot(v1,v2)
#         dot_list.append(dot)
        
#     k = int((N/2))
#     return eval_chebyt(k, cosphi)**2 - dot_list* eval_chebyu(k-1, cosphi)**2
    
def eval_chebyt_plot(t, N, Ax, Az ):
    t = 2*t
    B_vec = numpy.array([0,0,Bz])
    A_vec = numpy.array([Ax, 0, Az])
    
    w0 = numpy.linalg.norm(uN_13C*B_vec)
    w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
    n0 = uN_13C*B_vec/w0
    n1 = (uN_13C*B_vec + A_vec)/w1
    
    p0 = w0 * t / 2
    p1 = w1 * t / 2
    
    term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
    term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
    cosphi=  term_c - numpy.dot(n0, n1) * term_s
    
    k = int((N/2))
    return eval_chebyt(k, cosphi)**2
      
def eval_chebyu_plot(t, N, Ax, Az ):
    t = 2*t
    B_vec = numpy.array([0,0,Bz])
    A_vec = numpy.array([Ax, 0, Az])
    
    w0 = numpy.linalg.norm(uN_13C*B_vec)
    w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
    n0 = uN_13C*B_vec/w0
    n1 = (uN_13C*B_vec + A_vec)/w1
    
    p0 = w0 * t / 2
    p1 = w1 * t / 2
    
    term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
    term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
    cosphi=  term_c - numpy.dot(n0, n1) * term_s
    
    k = int((N/2))
    return eval_chebyu(k-1, cosphi)**2 
    
taus_lin = numpy.linspace(0, 10,600)



###____ include single 13C coupling ____###
# fit_func = lambda  t, Ax, Az: pop_S(t,num_pi_pulses, Ax, Az,0.09836647,  0.6, 0.04, 0, 
#                                                   dd_model_coeff_dict['{}'.format(num_pi_pulses)] ) 

fit_func = lambda  t,N, Ax, Az: (S_13C(t, N, Ax, Az) + 1)/2
A_amp = 1
A_ang = 0.2

fig, ax = plt.subplots()
Ax = A_amp*numpy.sin(A_ang)
Az = A_amp*numpy.cos(A_ang)
init_params = [6, Ax, Az,]

ax.plot(
        taus_lin,
        fit_func(taus_lin, *init_params),
        "-",
        color="red",
        label="XY4-6",
    ) 

init_params = [4, Ax, Az,]

ax.plot(
        taus_lin,
        fit_func(taus_lin, *init_params),
        "-",
        color="blue",
        label="XY4-4",
    ) 

 
init_params = [2, Ax, Az,]

ax.plot(
        taus_lin,
        fit_func(taus_lin, *init_params),
        "-",
        color="black",
        label="XY4-3",
    ) 

# ax.plot(
#         taus_lin,
#         eval_chebyt_plot(taus_lin, *init_params),
#         "-",
#         color="green",
#         label="T",
#     ) 

ax.set_xlabel(r"Inter-pulse time, $\tau$ (us)")
ax.set_ylabel("Coherence")

ax.legend()

