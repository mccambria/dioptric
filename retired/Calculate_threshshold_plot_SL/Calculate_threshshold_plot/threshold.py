# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:03:32 2020

@author: samli
"""



import scipy.stats
import scipy.special
import math  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import photonstatistics as model

#%% import data 

# nv0 = data.nv_0_counts
# nvm = data.nv_m_counts
# readout time in ms
readout_time = 300

#%% function 


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

# plot the probability distribution, array1 = nv0, array2 = nvm
def plot_dis(array1,array2):
    if len(array2) == 1:
        unique_value1, relative_freq1 = get_Probability_distribution(array1.tolist())
        plt.plot(unique_value1,relative_freq1,'-o')
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density") 
        plt.show()
        return True
         
    else: 
        unique_value1, relative_freq1 = get_Probability_distribution(array1.tolist())
        unique_value2, relative_freq2 = get_Probability_distribution(array2.tolist())
        plt.plot(unique_value1,relative_freq1,'-ro')
        plt.plot(unique_value2,relative_freq2,'-go')
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        plt.legend(("initialized with red","initialized with green"))
        plt.show()
        return True

def find_threshold(nv0,nvm):
    mean1 = int(np.mean(nv0))
    if mean1 == 0:
        mean1 = 1
    mean2 = int(np.mean(nvm))+1
    if mean1 > mean2 :
        print("Input wrong order or invalid data")
        return np.array([None, 0])
    #search the threshold inbetween the two means
    if mean2-mean1 == 1:
        thre_list = np.linspace(mean1,mean2,4*int((mean2-mean1)))
    if mean2 - mean1 == 0:
        return np.array([None, 0])
    else:
        thre_list = np.linspace(mean1,mean2,2*int((mean2-mean1)))
    #calculate the area above and below the threshold
    def get_area(alist, thred):
        unique_value, prob = get_Probability_distribution(alist)
        area_below = 0
        area_above = 0
        for i in range(len(unique_value)):
            if unique_value[i] < thred:
                area_below += prob[i]
            else: 
                area_above += prob[i]
        return [area_below,area_above]
    #calculate the probability difference btw two cases: if n >= n_thresh or n < n_thresh, successfully determine the state
    prob_diff_list = []
    for i in range(len(thre_list)):     
        thred = thre_list[i]
        area_below_nv0 = get_area(nv0.tolist(),thred)[0]
        area_below_nvm = get_area(nvm.tolist(),thred)[0]
        area_above_nv0 = get_area(nv0.tolist(),thred)[1]
        area_above_nvm = get_area(nvm.tolist(),thred)[1]
        prob_below = area_below_nv0 / (area_below_nvm + area_below_nv0)
        prob_above = area_above_nvm/ (area_above_nvm+ area_above_nv0)
        prob_diff_list.append(abs(prob_below - prob_above))

    thre_index = prob_diff_list.index(min(prob_diff_list))
    result_thresh = thre_list[thre_index]
    fidelity = get_area(nvm.tolist(), result_thresh)[1] / (get_area(nv0.tolist(), result_thresh)[1] + get_area(nvm.tolist(), result_thresh)[1])
    return np.array([result_thresh, fidelity])


# mu1 = mean of nv0 count distribution, mu2 = mean of nvm count distribution 
def calculate_threshold(mu1,mu2,x_data):
    threshold_list = np.linspace(int(mu1),int(mu2)+1,int(mu2 - mu1))
    dis1 = scipy.stats.poisson.pmf(x_data,mu1)
    dis2 = scipy.stats.poisson.pmf(x_data,mu2)
    prob_diff_list = []
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
        prob_below = area_below_nv0 / (area_below_nvm + area_below_nv0)
        prob_above = area_above_nvm/ (area_above_nvm+ area_above_nv0)
        prob_diff_list.append(abs(prob_below - prob_above))
    thre_index = prob_diff_list.index(min(prob_diff_list))
    result_thresh = threshold_list[thre_index]
    fidelity = get_area(x_data,dis2.tolist(), result_thresh)[1] / (get_area(x_data,dis1.tolist(), result_thresh)[1] + get_area(x_data,dis2.tolist(), result_thresh)[1])   
    return np.array([result_thresh, fidelity])

# mu1 = mean of nv0 count distribution, mu2 = mean of nvm count distribution 
def calculate_threshold2(readout_time,x_data,rate_fit):
    tR = readout_time
    threshold_list = np.linspace(int(mu1),int(mu2)+1,int(mu2) - int(mu1) +2)
    dis1 = np.array(model.get_PhotonNV0_list(x_data, tR, rate_fit, 1))
    dis2 = np.array(model.get_PhotonNVm_list(x_data, tR, rate_fit, 1))
    prob_diff_list = []
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
        prob_below = area_below_nv0 / (area_below_nvm + area_below_nv0)
        prob_above = area_above_nvm/ (area_above_nvm+ area_above_nv0)
        prob_diff_list.append(abs(prob_below - prob_above))
    thre_index = prob_diff_list.index(min(prob_diff_list))
    n_thresh = threshold_list[thre_index]
    fidelity1 = get_area(x_data,dis2.tolist(), n_thresh)[1] / (get_area(x_data,dis1.tolist(), n_thresh)[1] + get_area(x_data,dis2.tolist(), n_thresh)[1])   
    fidelity2 = get_area(x_data,dis1.tolist(), n_thresh)[0] / (get_area(x_data,dis1.tolist(), n_thresh)[0] + get_area(x_data,dis2.tolist(), n_thresh)[0])   
    return np.array([n_thresh,  0.5*(fidelity1 + fidelity2 )])

def fidelity(n_thresh,x_data,mu1,mu2):
    dis1 = scipy.stats.poisson.pmf(x_data,mu1)
    dis2 = scipy.stats.poisson.pmf(x_data,mu2)
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
    fidelity1 = get_area(x_data,dis2.tolist(), n_thresh)[1] / (get_area(x_data,dis1.tolist(), n_thresh)[1] + get_area(x_data,dis2.tolist(), n_thresh)[1])   
    fidelity2 = get_area(x_data,dis1.tolist(), n_thresh)[0] / (get_area(x_data,dis1.tolist(), n_thresh)[0] + get_area(x_data,dis2.tolist(), n_thresh)[0])   
    return 0.5*(fidelity1 + fidelity2 )

def fidelity2(n_thresh,readout_time,x_data,rate_fit):
    tR = readout_time
    dis1 = np.array(model.get_PhotonNV0_list(x_data, tR, rate_fit, 1))
    dis2 = np.array(model.get_PhotonNVm_list(x_data, tR, rate_fit, 1))
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
    fidelity1 = get_area(x_data,dis2.tolist(), n_thresh)[1] / (get_area(x_data,dis1.tolist(), n_thresh)[1] + get_area(x_data,dis2.tolist(), n_thresh)[1])   
    fidelity2 = get_area(x_data,dis1.tolist(), n_thresh)[0] / (get_area(x_data,dis1.tolist(), n_thresh)[0] + get_area(x_data,dis2.tolist(), n_thresh)[0])   
    return 0.5*(fidelity1 + fidelity2 )

def bi_poisson_model(x,mu1,mu2,A):
    B = 1 - A
    return A*scipy.stats.poisson.pmf(x,mu1) + B*scipy.stats.poisson.pmf(x,mu2)

def bi_poisson_model2(x,mu2,A):
    mu1 = 2.64
    B = 1 -A
    return A*scipy.stats.poisson.pmf(x,mu1) + B*scipy.stats.poisson.pmf(x,mu2)

def bi_poisson_model3(x,mu1,mu2,A,B):
    return A*scipy.stats.poisson.pmf(x,mu1) + B*scipy.stats.poisson.pmf(x,mu2)

def bi_poisson_plot(x_data,mu1,mu2,A,B):
    pois1 = bi_poisson_model3(x_data,mu1,mu2,A,0)
    pois2 = bi_poisson_model3(x_data,mu1,mu2,0,B)
    return pois1, pois2

#%% data analysis 12/02
# x_data = np.linspace(0,35,36)

# plot_dis(nv0,nvm)
# plot_dis(nv0,[0])
# plot_dis(nvm,[0])

# fit_0 = [ 2.65929876 ,10.96778535 , 0.79225763]
# fit_m = [ 3.61726212, 14.62414319,  0.42092031]

# u_value1, freq1 = get_Probability_distribution(nvm.tolist())
# popt,pcov = curve_fit(bi_poisson_model,u_value1,freq1,p0 = [2.64,10.3, 0.3])
# print(popt)
# print(np.diag(pcov))
# mu1,mu2,A = popt
# pois1, pois2 = bi_poisson_plot(x_data,mu1,mu2,A,1-A)
# print(calculate_threshold(mu1,mu2,x_data))
# fig1, ax = plt.subplots()
# ax.plot(u_value1,freq1,'-go')
# ax.plot(x_data,bi_poisson_model(x_data,mu1,mu2,A),color = "r")
# ax.plot(x_data,pois1,color = 'grey')
# ax.plot(x_data,pois2,color = 'k')
# textstr = '\n'.join((
#     r'$\mu_1=%.2f$' % (mu1, ),
#     r'$\mu_2=%.2f$'% (mu2, ),
#     r'$A =%.2f$'% (A, ),
#     r'$B =%.2f$'% (1-A, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# plt.show()


# u_value2, freq2 = get_Probability_distribution(nv0.tolist())
# popt,pcov = curve_fit(bi_poisson_model,u_value2,freq2,p0 = [2,13, 0.9])
# print(popt)
# print(np.diag(pcov))
# mu1,mu2,A = popt
# pois1, pois2 = bi_poisson_plot(x_data,mu1,mu2,A,1-A)
# print(calculate_threshold(mu1,mu2,x_data))
# fig2, ax = plt.subplots()
# ax.plot(u_value2,freq2,'-ro')
# ax.plot(x_data,bi_poisson_model(x_data,mu1,mu2,A),color = "orange")
# ax.plot(x_data,pois1,color = 'grey')
# ax.plot(x_data,pois2,color = 'k')
# textstr = '\n'.join((
#     r'$\mu_1=%.2f$' % (mu1, ),
#     r'$\mu_2=%.2f$'% (mu2, ),
#     r'$A =%.2f$'% (A, ),
#     r'$B =%.2f$'% (1-A, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# plt.show()



# print(fidelity(8,x_data,fit_m[0],fit_m[1]))
# print(fidelity(7,x_data,fit_0[0],fit_0[1]))



# popt,pcov = curve_fit(bi_poisson_model2,u_value1,freq1,p0 = [12, 0.3])
# print(popt)
# print(np.diag(pcov))
# mu2,A = popt
# mu1 = fit_0[0]
# pois1, pois2 = bi_poisson_plot(x_data,mu1,mu2,A,1-A)
# print(calculate_threshold(mu1,mu2,x_data))
# fig1, ax = plt.subplots()
# ax.plot(u_value1,freq1,'-go')
# ax.plot(x_data,bi_poisson_model(x_data,mu1,mu2,A),color = "r")
# ax.plot(x_data,pois1,color = 'grey')
# ax.plot(x_data,pois2,color = 'k')
# textstr = '\n'.join((
#     r'$\mu_1=%.2f$' % (mu1, ),
#     r'$\mu_2=%.2f$'% (mu2, ),
#     r'$A =%.2f$'% (A, ),
#     r'$B =%.2f$'% (1-A, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# plt.show()


# combined_count = nv0.tolist()+nvm.tolist()
# random.shuffle(combined_count)
# u_value2, freq2 = get_Probability_distribution(combined_count)
# popt,pcov = curve_fit(bi_poisson_model,u_value2,freq2,p0 = [2,13, 0.9])
# print(popt)
# print(np.diag(pcov))
# mu1,mu2,A = popt
# pois1, pois2 = bi_poisson_plot(x_data,mu1,mu2,A,1-A)
# print(calculate_threshold(mu1,mu2,x_data))
# fig2, ax = plt.subplots()
# ax.plot(u_value2,freq2,'-bo')
# ax.plot(x_data,bi_poisson_model(x_data,mu1,mu2,A),color = "orange")
# ax.plot(x_data,pois1,color = 'grey')
# ax.plot(x_data,pois2,color = 'k')
# textstr = '\n'.join((
#     r'$\mu_1=%.2f$' % (mu1, ),
#     r'$\mu_2=%.2f$'% (mu2, ),
#     r'$A =%.2f$'% (A, ),
#     r'$B =%.2f$'% (1-A, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# plt.show()


# fig3,ax = plt.subplots()
# ax.plot(x_data,0.5*scipy.stats.poisson.pmf(x_data,3),"-o")
# ax.plot(x_data,0.5*scipy.stats.poisson.pmf(x_data,14),"-o")
# plt.axvline(x=7,color = "red")
# textstr = '\n'.join((
#     r'$\mu_1=%.2f$' % (3, ),
#     r'$\mu_2=%.2f$'% (14, ),
#     r'$A =%.2f$'% (0.5, ),
#     r'$B =%.2f$'% (1-0.5, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# print(calculate_threshold(3,14,x_data))
# print(fidelity(7,x_data,3,14))

        
#%% Shields' model analysis fit to combined_counts
# guess =[ 0.01 ,0.1,1,0.15 ] 
# fit,dev = model.get_curve_fit2(20,0,combined_count ,guess)
# print(fit,np.diag(dev))
# fit = [0.00944542, 0.06967639, 0.9207457 , 0.12080523]
# fit = [0.01695567, 0.0655378 , 0.92056857, 0.12274083]
# curve = model.get_photon_distribution_curve(36,20,0.00944542, 0.06967639, 0.9207457 , 0.12080523, 0.5)
# u_value2, freq2 = get_Probability_distribution(combined_count)
# fig4, ax = plt.subplots()
# ax.plot(u_value2,freq2,"-o")
# ax.plot(x_data,curve)
# textstr = '\n'.join((
#     r'$g_0(s^{-1}) =%.2f$'% (9.4, ),
#     r'$g_1(s^{-1})  =%.2f$'% (70.0, ),
#     r'$\mu_0 =%.2f$'% (2.4, ),
#     r'$\mu_1 =%.2f$'% (18.4, )))
# ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")

#%% weight extraction 
# guess = [0.7]
# popt,pcov = model.get_curve_fit_to_weight(20,0,nvm.tolist(),guess,fit)
# print(popt)
# u_value1, freq1 = get_Probability_distribution(nvm.tolist())
# curve = model.get_photon_distribution_curve(36,20,0.00944542, 0.06967639, 0.9207457 , 0.12080523, popt[0])
# fig4, ax = plt.subplots()
# ax.plot(u_value1,freq1,"-go")
# ax.plot(x_data,curve)
# textstr = '\n'.join((
#     r'$g_0(s^{-1}) =%.2f$'% (9.4, ),
#     r'$g_1(s^{-1})  =%.2f$'% (70.0, ),
#     r'$\mu_0 =%.2f$'% (2.4, ),
#     r'$\mu_1 =%.2f$'% (18.4, ),
#     r'$A_1 =%.2f$'% (popt[0], )))
# ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")




# guess = [0.1]
# popt,pcov = model.get_curve_fit_to_weight(20,0,nv0.tolist(),guess,fit)
# popt,pcov = model.get_curve_fit_to_weight(20,0,nv0.tolist(),guess,fit)
# print(popt)
# u_value1, freq1 = get_Probability_distribution(nv0.tolist())
# curve = model.get_photon_distribution_curve(36,20,0.00944542, 0.06967639, 0.9207457 , 0.12080523, popt[0])
# fig4, ax = plt.subplots()
# ax.plot(u_value1,freq1,"-ro")
# ax.plot(x_data,curve)
# textstr = '\n'.join((
#     r'$g_0(s^{-1}) =%.2f$'% (9.4, ),
#     r'$g_1(s^{-1})  =%.2f$'% (70.0, ),
#     r'$\mu_0 =%.2f$'% (2.4, ),
#     r'$\mu_1 =%.2f$'% (18.4, ),
#     r'$A_1 =%.2f$'% (popt[0], )))
# ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
# print(popt)
        
#%% threshold calculation       
        

# nvm_curve = model.get_PhotonNVm_list(x_data,20,fit,0.5)
# nv0_curve = model.get_PhotonNV0_list(x_data,20,fit,0.5)
# u_value1, freq1 = get_Probability_distribution(combined_count)
# fig4, ax = plt.subplots()
# ax.plot(u_value1,freq1,"-o")
# ax.plot(x_data,curve)
# ax.plot(x_data,nvm_curve,"green")
# ax.plot(x_data,nv0_curve,"red")
# plt.axvline(x=5,color = "k",linestyle = ":")
# textstr = '\n'.join((
#     r'$n_{thresh} =%.2f$'% (5, ),
#     r'$fidelity  =%.2f$'% (0.82, )))
# ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
        
# print(calculate_threshold2(20,x_data.tolist(),fit))
        
# guess =  [0.005, 0.035, 0.6207457 , 0.07880523]
# fit = guess
# nvm_curve = model.get_PhotonNVm_list(x_data,10,fit,0.5)
# nv0_curve = model.get_PhotonNV0_list(x_data,10,fit,0.5)        
# fig5, ax = plt.subplots()
# n_thresh,fidelity = calculate_threshold2(10,x_data.tolist(),fit)
# ax.plot(x_data,nvm_curve,"green")
# ax.plot(x_data,nv0_curve,"red")
# plt.axvline(x= n_thresh,color = "k",linestyle = ":")
# textstr = '\n'.join((
#     r'$n_{thresh} =%.2f$'% (n_thresh, ),
#     r'$fidelity  =%.2f$'% (fidelity, )))
# ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# plt.xlabel("Number of counts")
# plt.ylabel("Probability Density")
            
        
        
        
        
        
        
        
        
        
        
        
    