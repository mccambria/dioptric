# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#photon statistics analysis module 
#all readout_time in unit of s
#all readout_power in unit of uW
import scipy.stats
import scipy.special
import math  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Define the powers of the yellow based on the nd_filter and aom voltage
# measured power is in uW

def measured_589_power(aom_volt, nd_filter):
    if nd_filter == 'nd_0':
        if aom_volt ==  0.1:
            measured_p = 3.19
        elif aom_volt ==  0.2:
            measured_p = 38.1
        elif aom_volt ==  0.3:
            measured_p = 125.1
        elif aom_volt ==  0.4:
            measured_p = 251
        elif aom_volt ==  0.5:
            measured_p = 335
        elif aom_volt ==  0.6:
            measured_p = 398
    if nd_filter == 'nd_0.5':
        if aom_volt ==  0.1:
            measured_p = 0.64
        elif aom_volt ==  0.2:
            measured_p = 10.98
        elif aom_volt ==  0.3:
            measured_p = 38.1
        elif aom_volt ==  0.4:
            measured_p = 71.5
        elif aom_volt ==  0.5:
            measured_p = 103.7
        elif aom_volt ==  0.6:
            measured_p = 125.5
    if nd_filter == 'nd_1.0':
        if aom_volt ==  0.1:
            measured_p = 0.1
        elif aom_volt ==  0.2:
            measured_p = 3.05
        elif aom_volt ==  0.3:
            measured_p = 11.09
        elif aom_volt ==  0.4:
            measured_p = 21.5
        elif aom_volt ==  0.5:
            measured_p = 31.3
        elif aom_volt ==  0.6:
            measured_p = 37.2
    if nd_filter == 'nd_1.5':
        if aom_volt ==  0.1:
            measured_p = 0.05
        elif aom_volt ==  0.2:
            measured_p = 0.27
        elif aom_volt ==  0.3:
            measured_p = 1.9
        elif aom_volt ==  0.4:
            measured_p = 4.2
        elif aom_volt ==  0.5:
            measured_p = 6.1
        elif aom_volt ==  0.6:
            measured_p = 7.4
            
    return measured_p
#%% define the four parameters that are used in the photon statistics
def get_g0(readout_power):
    P = readout_power*10**-3
    g0 = 39*P**2 /(1+P/134)
    return g0 

def get_g1(readout_power):
    P = readout_power*10**-3 
    g1 = 310*P**2 / (1 + P/53.2)
    return g1 

def get_y0(readout_power):
    P = readout_power*10**-3
    y0 = 1.63*10**3 * P/(1+P/134) + 0.268*10**3
    return y0 

def get_y1(readout_power):
    P = readout_power*10**-3
    y1 = 46.2*10**3 * P/(1+ P/53)+ 0.268*10**3
    return y1 

#%% Photon statistical model and fidelity
# Photon statistics of initializing in NV- and reading out for tR at power P
# return the probabilty of getting n photons
def get_Probability_distribution(aList):
    def get_unique_value(aList):
        unique_value_list = []
        for i in range(0,len(aList)):
            if aList[i] not in unique_value_list:
                unique_value_list.append(aList[i])
        unique_value_list.sort()
        return unique_value_list
    unique_value = get_unique_value(aList)
    relative_frequency = []
    for i in range(0,len(unique_value)):
        relative_frequency.append(aList.count(unique_value[i])/ (len(aList)))

    return unique_value, relative_frequency

def PhotonNVm(n,readout_time,readout_power, rate_fit):
    P = readout_power
    g0,g1,y1,y0 = rate_fit
    tR =readout_time 
    poisspdf2 = scipy.stats.poisson(y1*tR)
    def Integ(t):
        poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
        integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
        return integ
    val, err = scipy.integrate.quad(Integ,0,tR)
    result = val + math.e**(-g1*tR)*poisspdf2.pmf(n)
    return result
# Photon statistics of initializing in NV0 and reading out for tR at power P\
# return probability of getting n photons
def PhotonNV0(n,readout_time,readout_power,rate_fit):
    P = readout_power
    g0,g1,y1,y0 = rate_fit
    tR =readout_time 
    poisspdf2 = scipy.stats.poisson(y0*tR)
    def Integ(t):
        poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
        integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
        return integ
    val, err = scipy.integrate.quad(Integ,0,tR)
    result = val + math.e**(-g0*tR)*poisspdf2.pmf(n)
    return result
# draw a graph of photon distribution
def plotPhotonDistribution(readout_time,readout_power, NV_state,max_range):
    tR = readout_time
    P = readout_power
    #Plot photon distribution when initializing in NV-
    if NV_state == 1:
        NVm_probability_list = []
        for i in range(0,max_range):
            NVm_probability_list.append([PhotonNVm(i,tR,P)])
        photon_number = list(range(0,max_range))
        plt.plot(photon_number,NVm_probability_list,'b')
        plt.xlabel('number of photons')
        plt.ylabel('P(n)')
        return NVm_probability_list
    #Plot photon distribution when initializing in NV0
    if NV_state == 0:
        NV0_probability_list = []
        for i in range(0,max_range):
            NV0_probability_list.append([PhotonNV0(i,tR,P)])
        photon_number = list(range(0,max_range))
        plt.plot(photon_number,NV0_probability_list,'r')
        plt.xlabel('number of photons')
        plt.ylabel('P(n)')
        return NV0_probability_list
    else:
        print('NV_state is not specified')
#plot two distributions together 
#plotPhotonDistribution(820*10**-3,240*10**-6,1,20)
#plotPhotonDistribution(820*10**-3,240*10**-6,0,20)
#plt.show()    
        
def get_PhotonNVm_list(photon_number,readout_time,rate_fit,weight):
    tR =readout_time 
    A = weight
    i =0 
    P = 0
    Probability_list = []
    for i in range(len(photon_number)):
        n = photon_number[i] 
        result = A*(PhotonNVm(n,tR,P,rate_fit))
        Probability_list.append(result)
    return Probability_list

def get_PhotonNV0_list(photon_number,readout_time,rate_fit,weight):
    tR =readout_time 
    A = weight
    i =0 
    P = 0
    Probability_list = []
    for i in range(len(photon_number)):
        n =photon_number[i] 
        result = A*(PhotonNV0(n,tR,P,rate_fit))
        Probability_list.append(result)
    return Probability_list

#get measurement fidelity 
def get_readout_fidelity(readout_time, readout_power, NV_state,n_threshold):
    tR = readout_time
    P = readout_power 
    n = 0 
    Probability_NVm_les_nth = 0 
    Probability_NV0_les_nth = 0 
    for n in range(0,n_threshold):
        Probability_NVm_les_nth += PhotonNVm(n,tR,P)
        Probability_NV0_les_nth += PhotonNV0(n,tR,P)
    # fidelity of correctly determining NV0 state
    if NV_state == 0:
        fidelity = Probability_NV0_les_nth /(Probability_NV0_les_nth + Probability_NVm_les_nth)
        return fidelity
    # fidelity of correctly determining NV- state
    if NV_state == 1:
        fidelity = (1 - Probability_NVm_les_nth) /((1-Probability_NV0_les_nth) + (1 - Probability_NVm_les_nth))
        return fidelity
#%% curve fit to the photon distribution

def get_photon_distribution_curve(tR, photon_number,g0,g1,y1,y0):    
    poisspdf2 = scipy.stats.poisson(y1*tR)
    poisspdf3 = scipy.stats.poisson(y0*tR)
    photon_number = photon_number
    i = 0
    curve = []
    for i in range(len(photon_number)): 
        n = photon_number[i]
        def IntegNVm(t):
            poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
            integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
            return integ
        def IntegNV0(t):
            poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
            integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
            return integ
        valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
        valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
        A = 0.5
        B = 1- A
        result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)
        curve.append(result)
        i += 1
    return curve

# fit the model to the actual data and return the 4 rates: g0, g1, y1, y0
def get_curve_fit(readout_time,readout_power,data,guess):
    tR = readout_time
    P = readout_power
    initial_guess = guess
    uniq_val, freq = get_Probability_distribution(data)
    photon_number = uniq_val
    def get_photon_distribution_curve(photon_number,g0,g1,y1,y0):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = list(photon_number)
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
                return integ
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
                return integ
            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            A = 0.35
            B = 1- A
            result = (A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n))
            curve.append(result)
            i += 1
        return curve 
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number, freq, p0 = initial_guess,bounds=(0,np.inf))
    return popt, pcov

#A = 0.85
def get_curve_fit_green(readout_time,readout_power,data,guess):
    tR = readout_time
    P = readout_power
    initial_guess = guess
    uniq_val, freq = get_Probability_distribution(data)
    photon_number = uniq_val
    def get_photon_distribution_curve(photon_number,g0,g1,y1,y0):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = list(photon_number)
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
                return integ
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
                return integ
            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            A = 1
            B = 1- A
            result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number, freq, p0 = initial_guess,bounds=(0,np.inf))
    return popt, pcov

#A = 0.13
def get_curve_fit_red(readout_time,readout_power,data,guess):
    tR = readout_time
    P = readout_power
    initial_guess = guess
    uniq_val, freq = get_Probability_distribution(data)
    photon_number = uniq_val
    def get_photon_distribution_curve(photon_number,g0,g1,y1,y0):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = list(photon_number)
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
                return integ
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
                return integ
            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            A = 0.13
            B = 1- A
            result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number, freq, p0 = initial_guess,bounds=(0,np.inf))
    return popt, pcov




def get_photon_distribution_curve_weight(photon_number,readout_time, g0,g1,y1,y0,A):    
    tR = readout_time 
    poisspdf2 = scipy.stats.poisson(y1*tR)
    poisspdf3 = scipy.stats.poisson(y0*tR)
    i = 0
    curve = []
    for i in range(len(photon_number)): 
        n = photon_number[i]
        def IntegNVm(t):
            poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
            integ = g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n)
            return integ           
        def IntegNV0(t):
            poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
            integ = g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n)
            return integ
        valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
        valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
        B = 1 -A
        result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)      
        curve.append(result)
        i += 1
    return curve 

# calculate the weight of each charge state given a set of data, A = NVm, B = NV0
def get_curve_fit_to_weight(readout_time,readout_power,data,guess, rate_fit):
    tR = readout_time
    P = readout_power
    initial_guess = guess
    uniq_val, freq = get_Probability_distribution(data)
    photon_number = uniq_val
    g0,g1,y1,y0 = rate_fit
    def get_photon_distribution_curve_weight(photon_number,A ):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = list(photon_number)
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n)
                return integ
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = g0*math.e**((g0-g1)*t-g0*tR)*poisspdf4.pmf(n) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n)
                return integ
            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            B = 1- A
            result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    popt, pcov = curve_fit(get_photon_distribution_curve_weight, photon_number, freq, p0 = initial_guess,bounds = (0,1))
    return popt, pcov

#def get_curve_fit_NVm(readout_time,readout_power,data):
#    tR = readout_time
#    P = readout_power
#    initial_guess = [10,100,10000,1000]
#    def get_photon_distribution_curve(photon_number,g0,g1,y1,y0):    
#        poisspdf2 = scipy.stats.poisson(y1*tR)
#        photon_number = list(photon_number)
#        i = 0
#        curve = []
#        for i in range(len(photon_number)): 
#            n = i
#            def IntegNVm(t):
#                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
#                integ = g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n)
#                return integ
#
#            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
#            result = valNVm + math.e**(-g1*tR)*poisspdf2.pmf(n)
#            curve.append(result)
#            i += 1
#        return curve 
#    max_range = len(data)
#    photon_number = list(range(0,max_range))
#    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number, data, p0 = initial_guess)
#    return popt  

# fit the model to the actual data and return the 4 rates: g0, g1, y1, y0, A
def get_curve_fit_weight(readout_time,readout_power,data,guess):
    tR = readout_time
    P = readout_power
    initial_guess = guess
    uniq_val, freq = get_Probability_distribution(data)
    photon_number = uniq_val
    def get_photon_distribution_curve(photon_number,g0,g1,y1,y0,A):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = list(photon_number)
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = (g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt((g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf1.pmf(n))
                return integ
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = (g0*math.e**((g1-g0)*t-g1*tR)*poisspdf4.pmf(n)*scipy.special.jv(0, 2*math.sqrt(g1*g0*t*(tR-t))) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n))
                return integ
            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            B = 1- A
            result = A*valNVm + B*valNV0 + A*math.e**(-g1*tR)*poisspdf2.pmf(n) + B*math.e**(-g0*tR)*poisspdf3.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number, freq, p0 = initial_guess,bounds=(0,np.inf))
    return popt, pcov

#%% test module that simulates data to test the curve fit
#given readout time and power, return a list of probability of observing each number of photon
def get_PhotonNV_list(photon_number,readout_time,readout_power):
    P = readout_power
    tR =readout_time 
    i =0 
    Probability_list = []
    for i in range(len(photon_number)):
        n = i 
        result = 0.5*(PhotonNVm(n,tR,P) + PhotonNV0(n,tR,P))
        Probability_list.append(result)
    return Probability_list

# give a set of test data based on the model given by Shields' paper 
def get_fake_data(data,variation):
    i = 0 
    fake_data = []
    for i in range(len(data)):
        fake_data.append(abs(data[i] + np.random.randint(-1,2)* variation * np.random.random_sample()))
    return fake_data

# fit the model to the fake data generated
def run_test(photon_number_range, variation,readout_time,readout_power):
    tR = readout_time
    P = readout_power
    photon_number = list(range(photon_number_range))
    data = get_PhotonNV_list(photon_number, tR, P)
    fake_data = get_fake_data(data, variation)
    g0, g1, y1, y0= get_curve_fit(tR, P, fake_data)
    print( "At power = "+ str(P*10**3) + " nW" + " : "+ "Expected value: " + "g0 = " + str(get_g0(P)) +' , ' + "g1 = "+ str(get_g1(P))+ ',' + 
    "y1 = " + str(get_y1(P)) + ' ; ' + 'y0 =' + str(get_y0(P)))
    print("Actual value: " + "g0 = " + str(g0) +' , ' + "g1 = "+ str(g1)+ ',' + 
    "y1 = " + str(y1) + ' ; ' + 'y0 =' + str(y0))
    plt.plot(photon_number, fake_data,'b')
    plt.plot(photon_number, get_photon_distribution_curve(photon_number_range,tR, g0,g1,y1,y0),'r')
    plt.xlabel('number of photons')
    plt.ylabel('P(n)')
    plt.show()
    
#%% Power/n_threshold optimization 

def get_optimized_power(readout_time, power_range, n_threshold):
    tR = readout_time
    power_list = list(np.linspace(power_range[0],power_range[1],num = 1000))
    fidelity_list = []
    i = 0 
    for i in range(len(power_list)):
        fidelity_list.append(get_readout_fidelity(tR, power_list[i], 0, n_threshold))
        i+=1
    highest_fidelity_index = fidelity_list.index(max(fidelity_list))
    highest_fidelity = max(fidelity_list)
    optimized_power = power_list[highest_fidelity_index]
    return highest_fidelity, optimized_power
    
def get_fidelity_difference(readout_time, readout_power, n_threshold):
    return abs(get_readout_fidelity(readout_time, readout_power, 0, n_threshold) - get_readout_fidelity(readout_time, readout_power, 1, n_threshold))

def get_fidelity_list_power(readout_time, power_range, n_threshold):
    tR = readout_time
    power_list = list(np.linspace(power_range[0],power_range[1],num = 1000))
    i =0 
    fidelity_difference_list = []
    for i in range(len(power_list)):
        fidelity_difference_list.append(get_fidelity_difference(tR, power_list[i],n_threshold))
    return fidelity_difference_list
                        
def get_optimized_n_threshold(readout_time, power_range, n_threshold_list):
    tR = readout_time
    fidelity_difference_list = []
    i = 0
    for i in range(len(n_threshold_list)):
        n_th = n_threshold_list[i]
        fidelity_difference_list.append(min(get_fidelity_list_power(tR, power_range, n_th)))
    lower_fidelity_difference_index = fidelity_difference_list.index(min(fidelity_difference_list))
    optimized_n = n_threshold_list[lower_fidelity_difference_index]
    return optimized_n

#given a power_range, a two-element list, return the optimized n_threshold, power, highest fidelity
def get_optimized_fidelity(readout_time, power_range, n_threshold_list):
    tR = readout_time
    P = power_range
    optimized_n_threshold = get_optimized_n_threshold(tR, P, n_threshold_list)
    highest_fidelity, optimized_power = get_optimized_power(tR, P, optimized_n_threshold)
    return optimized_n_threshold, optimized_power, highest_fidelity

# given a power and readout time, find the optimized threshold 
def calculate_threshold(readout_time,x_data,rate_fit):
    tR = readout_time
    mu1 = rate_fit[3]*readout_time
    mu2 = rate_fit[2]*readout_time
    threshold_list = np.linspace(int(mu1),int(mu2)+2,int(mu2) - int(mu1) +3)
    dis1 = np.array(get_PhotonNV0_list(x_data, tR, rate_fit, 1))
    dis2 = np.array(get_PhotonNVm_list(x_data, tR, rate_fit, 1))
    fidelity_list = []
    def get_area(x_data,dis, thred):
        unique_value = x_data
        prob = dis
        area_below = 0
        area_above = 0
        for i in range(len(unique_value)):
            if unique_value[i] < thred:
                area_below += prob[i]
            else: 
                area_above += prob[i]
        return [area_below,area_above]
    for i in range(len(threshold_list.tolist())):
        thred = threshold_list[i]
        area_below_nv0 = get_area(x_data,dis1.tolist(),thred)[0]
        area_below_nvm = get_area(x_data,dis2.tolist(),thred)[0]
        area_above_nv0 = get_area(x_data,dis1.tolist(),thred)[1]
        area_above_nvm = get_area(x_data,dis2.tolist(),thred)[1]
        # prob_below = area_below_nv0 / (area_below_nvm + area_below_nv0)
        # prob_above = area_above_nvm/ (area_above_nvm+ area_above_nv0)
        # prob_diff_list.append(abs(prob_below - prob_above))
        pnvm_c = area_above_nvm
        pnv0_c = area_below_nv0
        fidelity_list.append( (pnvm_c + pnv0_c)/2 )
    thre_index = fidelity_list.index(max(fidelity_list))
    n_thresh = threshold_list[thre_index]
    # fidelity1 = get_area(x_data,dis2.tolist(), n_thresh)[1] / (get_area(x_data,dis1.tolist(), n_thresh)[1] + get_area(x_data,dis2.tolist(), n_thresh)[1])   
    # fidelity2 = get_area(x_data,dis1.tolist(), n_thresh)[0] / (get_area(x_data,dis1.tolist(), n_thresh)[0] + get_area(x_data,dis2.tolist(), n_thresh)[0])   
    return np.array([n_thresh,  max(fidelity_list)])

#nv_para = array([g0(P),g1(P),y0(P),y1(P)]) =  [A0,B0,A1,B1,C0,D0,C1,D1]
def optimize_single_shot_readout(power_range,time_range,nv_para,optimize_steps):
    m = optimize_steps
    power_array = np.linspace(power_range[0],power_range[1],m)
    time_array = np.linspace(time_range[0],time_range[1],m)
    Ag0, Bg0, Ag1, Bg1,Ay0, By0, Ay1, By1 = nv_para
    x_data = np.linspace(0,60,61)
    def rate_model(x,A,B):
        return A*x**2 + B*x
    def FL_model(x,A, B):
        return A*x + B
    fid_power_list = []
    time_power_list = []
    thresh_power_list = []
    for i in range(0,m):
        fid_time_list = []
        thresh_time_list = []
        power = power_array[i]
        #fit = [g0,g1,y1,y0]
        fit = [10**-3*rate_model(power,Ag0,Bg0),10**-3*rate_model(power,Ag1,Bg1),FL_model(power,Ay1,By1),FL_model(power,Ay0,By0)]
        for j in range(0,m):
            tR = time_array[j]
            n_th,fid = calculate_threshold(tR,x_data,fit)
            fid_time_list.append(fid)
            thresh_time_list.append(n_th)
        max_fid = max(fid_time_list)
        index = fid_time_list.index(max_fid)
        max_fid_time = time_array[index]
        max_nth_time = thresh_time_list[index]
        fid_power_list.append(max_fid)
        time_power_list.append(max_fid_time)
        thresh_power_list.append(max_nth_time)
    max_fid = max(fid_power_list)
    index = fid_power_list.index(max_fid)
    max_fid_time = time_power_list[index]
    max_fid_power = power_array[index]
    max_fid_nth = thresh_power_list[index]
    return np.array([max_fid_time,max_fid_power,max_fid_nth,max_fid])
            
        
#%% curve fit and figure drawing for each charge state
#given actural data: 
#unique_value: number of photons that are collected; 
#relative_frequency: the probabiltiy of appearance for each number of photons
def get_curve_fit_NVm(readout_time,readout_power,unique_value, relative_frequency):
    tR = readout_time
    P = readout_power
    initial_guess = [10,100,10000,1000]
    def get_photon_distribution_curve(unique_value,g0,g1,y1,y0):    
        poisspdf2 = scipy.stats.poisson(y1*tR)
        photon_number = unique_value
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNVm(t):
                poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
                integ = g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n) + math.sqrt(abs(g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(abs(g1*g0*t*(tR-t))))* poisspdf1.pmf(n)
                return integ

            valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
            result = valNVm + math.e**(-g1*tR)*poisspdf2.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    photon_number =unique_value
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number,  relative_frequency, p0 = initial_guess)
    return popt 

def get_photon_distribution_curveNVm(photon_number,readout_time, g0,g1,y1,y0):  
    if g0 < 0:
            g0 = 0
    if g1 < 0:
        g1 = 0
    tR = readout_time 
    poisspdf2 = scipy.stats.poisson(y1*tR)
    i = 0
    curve = []
    photon_number_list = list(range(photon_number))
    for i in range(len(photon_number_list)): 
        n = i
        def IntegNVm(t):
            poisspdf1 = scipy.stats.poisson(y1*t+y0*(tR-t))
            integ = g1*math.e**((g0-g1)*t-g0*tR)*poisspdf1.pmf(n)
            + math.sqrt(abs(g1*g0*t/(tR-t)))*math.e**((g0-g1)*t-g0*tR)*scipy.special.jv(1, 2*math.sqrt(abs(g1*g0*t*(tR-t))))* poisspdf1.pmf(n)
            return integ
        valNVm, err = scipy.integrate.quad(IntegNVm,0,tR)
        result = valNVm + math.e**(-g1*tR)*poisspdf2.pmf(n) 
        curve.append(result)
        i += 1
    return curve 

def get_curve_fit_NV0(readout_time,readout_power,unique_value, relative_frequency):
    tR = readout_time
    P = readout_power
    initial_guess = [10,100,10000,1000]
    def get_photon_distribution_curve(unique_value,g0,g1,y1,y0):
        poisspdf3 = scipy.stats.poisson(y0*tR)
        photon_number = unique_value
        i = 0
        curve = []
        for i in range(len(photon_number)): 
            n = photon_number[i]
            def IntegNV0(t):
                poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
                integ = g0*math.e**((g0-g1)*t-g0*tR)*poisspdf4.pmf(n) + math.sqrt(abs(g1*g0*t/(tR-t)))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(abs(g1*g0*t*(tR-t))))* poisspdf4.pmf(n)
                return integ

            valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
            result = valNV0 + math.e**(-g0*tR)*poisspdf3.pmf(n)
            curve.append(result)
            i += 1
        return curve 
    photon_number =unique_value
    popt, pcov = curve_fit(get_photon_distribution_curve, photon_number,  relative_frequency, p0 = initial_guess)
    return popt 

def get_photon_distribution_curveNV0(photon_number,readout_time, g0,g1,y1,y0):    
    if g0 < 0:
        g0 = 0
    if g1 < 0:
        g1 = 0
    tR = readout_time 
    poisspdf3 = scipy.stats.poisson(y0*tR)
    i = 0
    curve = []
    photon_number_list = list(range(photon_number))
    for i in range(len(photon_number_list)): 
        n = i
        def IntegNV0(t):
            poisspdf4 = scipy.stats.poisson(y0*t+y1*(tR-t))
            integ = g0*math.e**((g0-g1)*t-g0*tR)*poisspdf4.pmf(n) + math.sqrt(g1*g0*t/(tR-t))*math.e**((g1-g0)*t-g1*tR)*scipy.special.jv(1, 2*math.sqrt(g1*g0*t*(tR-t)))* poisspdf4.pmf(n)
            return integ

        valNV0, err = scipy.integrate.quad(IntegNV0,0,tR)
        result = valNV0 + math.e**(-g0*tR)*poisspdf3.pmf(n)
        curve.append(result)
        i += 1
    return curve    
#%% quick single poisson curve fit 
def get_sigle_poisson_distribution_fit(readout_time,readout_power,unique_value, relative_frequency):
    tR = readout_time
    number_of_photons = unique_value
    def PoissonDistribution(number_of_photons, F):
        poissonian =[]
        for i in range(len(number_of_photons)):
            n = number_of_photons[i]
            poissonian.append(((F*tR)**n) * (math.e ** (-F*tR)) /math.factorial(n))
        return poissonian
    popt, pcov = curve_fit(PoissonDistribution, number_of_photons,  relative_frequency)
    return popt

def get_single_poisson_distribution_curve(number_of_photons,readout_time, F):
    poissonian_curve =[]
    tR = readout_time
    for i in range(len(number_of_photons)):
        n = number_of_photons[i]
        poissonian_curve.append(((F*tR)**n) * (math.e ** (-F*tR)) /math.factorial(n))
    return poissonian_curve   

#%% quick double poisson curve fit 
def get_poisson_distribution_fit(readout_time,unique_value, relative_frequency):
    tR = readout_time
    number_of_photons = unique_value
    def PoissonDistribution(number_of_photons, a, b, numbla1, numbla2):
        #numbla1 and numbla2 represent the fluorescence rate 
        poissonian =[]
        for i in range(len(number_of_photons)):
            n = number_of_photons[i]
            poissonian.append((a*(numbla1*tR)**n) * (math.e ** (-numbla1*tR)) /math.factorial(n) + b*((numbla2*tR)**n) * (math.e ** (-numbla2*tR)) /math.factorial(n))
        return poissonian
    popt, pcov = curve_fit(PoissonDistribution, number_of_photons,  relative_frequency)
    return popt

def get_poisson_distribution_curve(number_of_photons,readout_time, a, b, numbla1, numbla2):
    poissonian_curve =[]
    tR = readout_time
    for i in range(len(number_of_photons)):
        n = number_of_photons[i]
        poissonian_curve.append((a*(numbla1*tR)**n) * (math.e ** (-numbla1*tR)) /math.factorial(n) + b*((numbla2*tR)**n) * (math.e ** (-numbla2*tR)) /math.factorial(n))
    return poissonian_curve 
#%% gaussian fit
def get_gaussian_distribution_fit(readout_time,readout_power,unique_value, relative_frequency):
    tR = readout_time
    number_of_photons = unique_value
    average_photon_number = 0
    for i in range(len(unique_value)):
        average_photon_number += relative_frequency[i] * unique_value[i]
    variance = 0 
    for i in range(len(unique_value)):
        variance += relative_frequency[i] * (unique_value[i])**2
    variance = variance - average_photon_number**2
    sigma_guess = math.sqrt(variance)
    def GaussianDistribution(number_of_photons,u,sigma, offset, coeff):
        gaussian = []
        for i in range(len(number_of_photons)):
            n = number_of_photons[i]
            gaussian.append(offset + coeff * math.e ** (-0.5*((n - u)/sigma)**2))
        return gaussian
    popt, pcov = curve_fit(GaussianDistribution,number_of_photons,relative_frequency,p0= [average_photon_number,sigma_guess,0,0.1])
    return popt

def get_gaussian_distribution_curve(number_of_photons,readout_time,u, sigma, offset, coeff):
    gaussian =[]
    for i in range(len(number_of_photons)):
        n = number_of_photons[i]
        gaussian.append(offset + coeff * math.e**(-0.5*((n - u)/sigma)**2))
    return gaussian

#%% photon counts monitoring module 
    
def get_time_axe(sequence_time, readout_time, photon_number_list):
    time_data = []
    for i in range(1,len(photon_number_list)+1):
        time_data.append(sequence_time*i)
    return time_data

def get_photon_counts(readout_time, photon_number_list):
    photon_counts = np.array(photon_number_list)/np.array(readout_time)
    return photon_counts.tolist()
        
        
