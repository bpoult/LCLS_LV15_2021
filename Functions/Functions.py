import pandas as pd 

import psana as ps
import numpy as np
import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as npply
import sys
import os
import h5py
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter as gf
from sklearn.utils import resample
sys.path.append('/reg/data/ana16/rix/rixlv1519/results/LCLS_LV15_2021/TestCode/EarlyScience/AnalyzeH5/')
from chemRIXSAnalysis import *
from ChemRIXSClasses import *
from filterTools import *
from raw_data_class import RawData as RDC
from pro_data_class import ProData as PDC

font = {'size'   : 16}
mpl.rc('font', **font)

cmap = plt.cm.get_cmap('terrain').reversed()

def process_fim(fim_number,channels,baseline_roi,signal_roi):
    fim_raw = fim_number
    baselines = np.asarray([fim_raw[:,i,baseline_roi] for i in channels])
    baselines = np.mean(baselines,2)
    
    specs = np.asarray([fim_raw[:,i,signal_roi] for i in channels])
    intensities = np.zeros(specs.shape)
    for i in range(0,specs.shape[0]):
        for j in range(0,specs.shape[1]):
            intensities[i,j,:]=specs[i,j,:]-baselines[i,j]
    intensities = -np.sum(intensities,2)

    return intensities, specs, baselines

def process_fim_2(fim_number,channels):
    fim_raw = fim_number
    specs = np.asarray([fim_raw[:,i,:] for i in channels])
    intensities = np.sum(specs,2)
    intensities = intensities

    return intensities

def sort_pump_v_unpump(laser,variable):
    pumped = variable[laser==1]
    unpumped = variable[laser ==0]
    return pumped, unpumped

def apply_filter(condition,variable):
    filtered = variable[condition]
    return filtered

def process_andor(andor_type,baseline_roi,signal_roi):
    andor = andor_type
    baselines = andor[:,baseline_roi]
    baselines = np.mean(baselines,1)
    
    specs = andor[:,signal_roi]
    intensities = np.zeros(specs.shape)
    for i in range(0,specs.shape[0]):
            intensities[i,:]=specs[i,:]-baselines[i]
    intensities = np.sum(intensities,1)

    return intensities, specs, baselines


def mono_spectrum(mono_encoder_ev, n_bins, y_vals):
    bin_centers = np.linspace(np.min(mono_encoder_ev),np.max(mono_encoder_ev),n_bins)
    bin_widths = (bin_centers[1]-bin_centers[0])/2
    
    intensity = np.zeros(n_bins)
    for i in range(0,len(bin_centers)):
        upper_bound = mono_encoder_ev<=bin_centers[i]+bin_widths
        lower_bound = mono_encoder_ev>=bin_centers[i]-bin_widths
        binning = np.logical_and(upper_bound,lower_bound)
        intensity[i] = np.mean(y_vals[binning])
    energy = bin_centers
    return energy,intensity

def energy_binning(x_value,y_value,n_bins):
    vernier_energy = np.squeeze(x_value)
    bins = np.linspace(np.min(vernier_energy),np.max(vernier_energy),n_bins)
    intensity=[]
    shots = []
    for i in range(0,len(bins)-1):
        bin_cond = np.logical_and(vernier_energy>=bins[i],vernier_energy<=bins[i+1])
        shots.append(y_value[bin_cond])
        intensity.append(np.mean(y_value[bin_cond]))
    intensity = np.asarray(intensity)
    shots = np.asarray(shots,dtype=object)
    return bins,intensity,shots


def bin_filtering(x_to_fit,y_to_fit,fim_chan,acceptance):
    filter_params = []
    for i in range(0,len(x_to_fit)):
        y = y_to_fit[i]
        x = x_to_fit[i][:,fim_chan]
        scale = np.max([x,y])
        poly_fit = np.polyfit(x, y,2)
        cond_poly_high = y < (x**2) * poly_fit[0] + x*poly_fit[1]+poly_fit[2] + acceptance*scale
        cond_poly_low = y > (x**2) * poly_fit[0] + x*poly_fit[1]+poly_fit[2] - acceptance*scale
        condition = cond_poly_high & cond_poly_low
        filter_params.append(condition)
    filter_params = np.asarray(filter_params,dtype=object)
    return filter_params



# def bounds_filter(data,parameters):
#     condition1 = data > parameters[0]
#     condition2 = data < parameters[1]
#     condition = np.logical_and(condition1,condition2)
#     return condition
    
    
def bounds_filter(raw_data,filt_param,plot_on):
    data = raw_data
    lower_bound = filt_param[0][0]
    upper_bound = filt_param[0][1]
    num_stds = filt_param[1][0]
    cond_low = data > np.nanmedian(data) - np.nanstd(data)*num_stds
    cond_high = data < np.nanmedian(data) + np.nanstd(data)*num_stds
    cond_abs_min = data > lower_bound
    if not upper_bound is 'None':
        cond_abs_max = data < upper_bound
    else:
        cond_abs_max = True
    condition = cond_low & cond_high & cond_abs_min & cond_abs_max
    if plot_on:
        plt.figure()
        _, bins, _ = plt.hist(raw_data, 100, label='unfiltered')
        _ = plt.hist(raw_data[condition], bins, rwidth=.5, label='filtered')
        plt.legend()
        plt.show()
    return condition
    
def lin_filter(data_1,data_2,filt_param,plot_on):
    var_x = data_1
    var_y = data_2
    scale = np.max([var_x,var_y])
    if filt_param[1]:
        m, _, _, _ = np.linalg.lstsq(var_x[:,np.newaxis],var_y,rcond=None)
        cond_lin_high = var_y < var_x * m + filt_param[0]*scale
        cond_lin_low = var_y > var_x * m - filt_param[0]*scale
        condition = cond_lin_high & cond_lin_low
    else:
        var_x = var_x
        var_y = var_y
        lin_fit = np.polyfit(var_x, var_y, 1)
        cond_lin_high = var_y < var_x * lin_fit[0] + lin_fit[1] + filt_param[0]*scale
        cond_lin_low = var_y > var_x * lin_fit[0] + lin_fit[1] - filt_param[0]*scale
        condition = cond_lin_high & cond_lin_low
    if plot_on:
        plt.figure()
        plt.scatter(var_x,var_y,alpha=0.95)
        plt.scatter(var_x[condition],var_y[condition],alpha=0.5)
        plt.show()

    return condition
    
    
def mono_calib(energy_raw,mono_encoder):
    NaN_entries = []
    energy_raw_new = []
    for i in range(len(energy_raw)):
        if np.isnan(energy_raw[i]):
            NaN_entries.append(i)
        else:
            energy_raw_new.append(energy_raw[i])
    NaN_entries = np.asarray(NaN_entries)
    mono_encoder_new = mono_encoder.copy()
    for i in range(len(NaN_entries)):
        mono_encoder_new[NaN_entries[i]] = 0
    mono_encoder_new = mono_encoder_new[mono_encoder_new!=0]
    monoCalib=npply.fit(mono_encoder_new,energy_raw_new,1)
    fit_coef = monoCalib.convert().coef
    mono_encoder_ev = []
    for i in range(len(mono_encoder)):
        mono_encoder_ev.append(fit_coef[0]+fit_coef[1]*mono_encoder[i])
    mono_encoder_ev = np.asarray(mono_encoder_ev)
    return mono_encoder_ev
  
def time_scan(lxt, andor_intensities):
    condition = True
    lxt_filt = lxt[condition]
    andor_intensities_filt = andor_intensities[condition]
    lxt_nonan = np.squeeze(myround(lxt_filt[np.logical_not(np.isnan(lxt_filt))],0.00001))
    lxt_u = np.unique((np.asarray(lxt_nonan)))
    andor_intensities_new = andor_intensities_filt[np.logical_not(np.isnan(lxt_filt))]
    if (len(lxt_u) == 1):
        d_bins = lxt_u[0]
        n_d_bins = 1
        andor_intensities_time = np.mean(andor_intensities_new)
        return d_bins, andor_intensities_time
    else:
        dstep = lxt_u[1]-lxt_u[0]
        d_bins = np.arange((lxt_u[0]-(dstep/2)),(lxt_u[-1]+(1*dstep/2)),dstep)
        n_d_bins = len(d_bins)
                      
    time_tracker = np.zeros([n_d_bins,len(lxt_nonan)])

    for i in range (len(lxt_nonan)):
        for j in range (n_d_bins):
            if j==0 & (lxt_nonan[i] <= d_bins[j] + dstep/2):
                time_tracker[j][i] = andor_intensities_new[i]
            elif j == n_d_bins - 1 & (lxt_nonan[i] >= d_bins[j] - dstep/2):
                time_tracker[j][i] = andor_intensities_new[i]
            elif (lxt_nonan[i] >= d_bins[j] - dstep/2) & (lxt_nonan[i] <= d_bins[j] + dstep/2):
                time_tracker[j][i] = andor_intensities_new[i]

    andor_intensities_time = np.zeros(n_d_bins)
        
    for i in range (n_d_bins):
        andor_intensities_time[i] = np.sum(time_tracker[i]) / np.count_nonzero(time_tracker[i])
                      
    return d_bins, andor_intensities_time
    
def process_data (raw_datas,filter_params,scan_type,scan_detector=7,norm_detector=4):
    
    pro_datas = []
    for i in range(0,len(raw_datas)):
        pro_data = PDC()
        pumped = raw_datas[i].laser==1
        unpumped = raw_datas[i].laser==0
        condition = filter_params[i].condition

        norm_by = raw_datas[i].I0_intensities_fim0[norm_detector,:]

        if scan_type is 'mono':

            n_bins = 100
            mono_encoder_ev = np.squeeze(mono_calib(raw_datas[i].energy_raw,raw_datas[i].mono_encoder))

            x_vals = mono_encoder_ev
            y_vals = raw_datas[i].intensities_fim2[scan_detector,:]

            ##### pumped #####
            energy,intensity_raw_pumped = \
            mono_spectrum(x_vals[pumped],n_bins,y_vals[pumped])

            energy,intensity_filtered_pumped = \
            mono_spectrum(x_vals[condition&pumped],n_bins,y_vals[condition&pumped])

            energy,intensity_norm_filtered_pumped = \
            mono_spectrum(x_vals[condition&pumped],n_bins,y_vals[condition&pumped]/norm_by[condition&pumped])

            ##### unpumped #####
            energy,intensity_raw_unpumped = \
            mono_spectrum(x_vals[unpumped],n_bins,y_vals[unpumped])

            energy,intensity_filtered_unpumped = \
            mono_spectrum(x_vals[condition&unpumped],n_bins,y_vals[condition&unpumped])

            energy,intensity_norm_filtered_unpumped = \
            mono_spectrum(x_vals[condition&unpumped],n_bins,y_vals[condition&unpumped]/norm_by[condition&unpumped])


            pro_data.changeValue(energy=energy,
                                 intensity_raw_pumped = intensity_raw_pumped,
                                 intensity_filtered_pumped = intensity_filtered_pumped,
                                 intensity_norm_filtered_pumped = intensity_norm_filtered_pumped,
                                 intensity_raw_unpumped = intensity_raw_unpumped,
                                 intensity_filtered_unpumped = intensity_filtered_unpumped,
                                 intensity_norm_filtered_unpumped = intensity_norm_filtered_unpumped)

            pro_datas = pro_datas + [pro_data]

        if scan_type is 'time':

            x_vals = raw_datas[i].lxt

            y_vals = raw_datas[i].andor_dir_intensities

            #pumped
            d_bins_raw, andor_intensities_time_raw = time_scan(x_vals,y_vals)

            d_bins_filt, andor_intensities_time_filt = time_scan(x_vals[condition],y_vals[condition])

            andor_intensities_time_norm_filt = \
            time_scan(x_vals[condition&unpumped],y_vals[condition&unpumped]/norm_by[condition&unpumped])[1]

            #unpumped
            andor_intensities_time_raw_pumped = time_scan(x_vals[pumped],y_vals[pumped])[1]

            andor_intensities_time_filt_pumped = time_scan(x_vals[condition&pumped],y_vals[condition&pumped])[1]

            andor_intensities_time_norm_filt_pumped = \
            time_scan(x_vals[condition&pumped],y_vals[condition&pumped]/norm_by[condition&pumped])[1]

            pro_data.changeValue(d_bins_raw=d_bins_raw,
                                 andor_intensities_time_raw=andor_intensities_time_raw,
                                 andor_intensities_time_filt=andor_intensities_time_filt,
                                 d_bins_filt=d_bins_filt,
                                 andor_intensities_time_norm_filt = andor_intensities_time_norm_filt,
                                 andor_intensities_time_raw_pumped = andor_intensities_time_raw_pumped,
                                 andor_intensities_time_filt_pumped = andor_intensities_time_filt_pumped,
                                 andor_intensities_time_norm_filt_pumped =andor_intensities_time_norm_filt_pumped
                                )

            pro_datas = pro_datas + [pro_data]
    
    print(scan_type+' has '+'been '+'completed')
    
    return pro_datas
    
def filter_param(raw_datas,n_fim,filter_detector,B_lowerBond = 4000,B_upperBond = 100000,L_condition = 0.05):
    filter_params = []
    for raw in raw_datas:
        filter_param = PDC()
        if n_fim == 0:
            B_condition_1 = bounds_filter(raw.I0_intensities_fim0[filter_detector,:],[[B_lowerBond,B_upperBond],[3]],False)
            L_condition_1 = lin_filter(raw.I0_intensities_fim0[filter_detector,:], raw.andor_dir_intensities,[L_condition,False],False)
        if n_fim == 1:
            B_condition_1 = bounds_filter(raw.I0_intensities_fim1[filter_detector,:],[[B_lowerBond,B_upperBond],[3]],False)
            L_condition_1 = lin_filter(raw.I0_intensities_fim1[filter_detector,:], raw.andor_dir_intensities,[L_condition,False],False)
        if n_fim == 2:
            B_condition_1 = bounds_filter(raw.I0_intensities_fim2[filter_detector,:],[[B_lowerBond,B_upperBond],[3]],False)
            L_condition_1 = lin_filter(raw.I0_intensities_fim2[filter_detector,:], raw.andor_dir_intensities,[L_condition,False],False)
        # raw.I0_intensities_fim0[4,:] is I0 intensities from fim0 channel 4


        bounds_condition = B_condition_1

        linearity_condition = L_condition_1
        
        condition = bounds_condition & linearity_condition

        filter_param.changeValue(condition=condition)
        filter_params = filter_params + [filter_param]
    return filter_params    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    