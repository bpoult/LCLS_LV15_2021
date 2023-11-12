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
sys.path.append('/reg/data/ana16/rix/rixlv1519/results/LCLS_LV15_2021/Functions/')
from Functions import *
from raw_data_class import RawData as RDC
from raw_class_2 import RawData_2 as RDC_2


font = {'size'   : 16}
mpl.rc('font', **font)

cmap = plt.cm.get_cmap('terrain').reversed()

def load_scans(small_data_folder,exp,scan):
    
    RawData = RDC()
    raw = h5py.File(small_data_folder+'%s_Run%04d.h5' % (exp,scan))
    events = np.array(raw['timestamp'])
    
    xgmd = np.array(raw['xgmd']['energy'])
    gmd = np.array(raw['gmd']['energy'])
    energy_raw = np.array(raw['epicsAll']['MONO_ENERGY_EV'])
    mono_encoder = np.array(raw['mono_encoder']['value'])
    pitch_raw = np.array(raw['epicsAll']['MONO_GRATING_PITCH'])
    horz_raw = np.array(raw['epicsAll']['MR3K4_pitch'])
    evrs = np.array(raw['timing']['eventcodes'])
    lxt = np.array(raw['epicsAll']['LAS_VIT_TIME'])
    laser = np.array(raw['lightStatus']['laser'])
    xray = np.array(raw['lightStatus']['xray'])

    
    nan_cond_1 = np.logical_not(np.isnan(np.squeeze(mono_encoder)))
    nan_cond_2 = np.logical_not(np.isnan(energy_raw))
    nan_cond_3 = np.logical_not(np.isnan(lxt))
    nan_cond_4 = np.logical_not(np.isnan(xgmd))
    nan_cond_5 = np.logical_not(np.isnan(gmd))
    nan_cond_6 = np.logical_not(np.isnan(laser))
    nan_cond_7 = np.logical_not(xray==0)

    nan_cond = nan_cond_1 & nan_cond_2 & nan_cond_3 & nan_cond_4 & nan_cond_5 & nan_cond_6 & nan_cond_7
    
    RawData.changeValue(events = events[nan_cond],
                            xgmd = xgmd[nan_cond],
                            gmd = gmd[nan_cond],
                            energy_raw = energy_raw[nan_cond],
                            mono_encoder = np.squeeze(mono_encoder)[nan_cond],
                            pitch_raw = pitch_raw[nan_cond],
                            horz_raw = horz_raw[nan_cond],
                            evrs = evrs[nan_cond],
                            lxt = lxt[nan_cond],
                            laser = laser[nan_cond],
                            nan_cond = nan_cond)
    
    try:
        mono_encoder_ev = []
        mono_encoder_ev = mono_calib(RawData.energy_raw,RawData.mono_encoder)
        print('generating mono_encoder_ev')
        RawData.changeValue(mono_encoder_ev = mono_encoder_ev)
        
    except:
        print('no mono_encoder')
    
    try:
        tt_pos=raw['tt']['fltpos'][:]
        print('loading TT')
        tt_posps=raw['tt']['fltpos_ps'][:]
        tt_posfwhm=raw['tt']['fltposfwhm'][:]
        
        RawData.changeValue(tt_pos = tt_pos[nan_cond],
                            tt_posps = tt_posps[nan_cond],
                            tt_posfwhm = tt_posfwhm[nan_cond])        
    except:
        print('no TT')
        TT = False

    try:
        print('loading fim0')
        fim0_raw = []
        for i in raw['rix_fim0_raw']:
            fim0_raw.append(np.array((raw['rix_fim0_raw'][i])))
        fim0_raw = np.moveaxis(np.asarray(fim0_raw),0,1)
        RawData.changeValue(fim0_raw = fim0_raw[nan_cond,:,:])
    except:
        print('no fim0')
        Fim0 = False

    try:
        print('loading fim1')
        fim1_raw = []
        for i in raw['rix_fim1_raw']:
            fim1_raw.append(np.array((raw['rix_fim1_raw'][i])))
        fim1_raw = np.moveaxis(np.asarray(fim1_raw),0,1)
        RawData.changeValue(fim1_raw = fim1_raw[nan_cond,:,:])

    except:
        print('no fim1')
        Fim1 = False

    try:
        andor_dir_raw = []
        print('loading andor')
        andor_dir_raw = np.array(raw['andor_dir']['full_area'])
        RawData.changeValue(andor_dir_raw = andor_dir_raw[nan_cond])

    except:
        print('no andor')
        Andor = False

    try:
        print('loading fim2')
        fim2_raw = []
        for i in raw['rix_fim2_raw']:
            fim2_raw.append(np.array((raw['rix_fim2_raw'][i])))
        fim2_raw = np.moveaxis(np.asarray(fim2_raw),0,1)
        RawData.changeValue(fim2_raw = fim2_raw[nan_cond,:,:])
        
    except:
        print('no fim2')
        Fim2 = False
    
    return RawData
    
    
def load_scans_2(small_data_folder,exp,scan):
    
    RawData = RDC()
    raw = h5py.File(small_data_folder+'%s_Run%04d.h5' % (exp,scan))
    
    xgmd = np.array(raw['xgmd']['energy'])
    gmd = np.array(raw['gmd']['energy'])
    energy_raw = np.array(raw['epicsAll']['MONO_ENERGY_EV'])
    mono_encoder = np.array(raw['mono_encoder']['value'])
    pitch_raw = np.array(raw['epicsAll']['MONO_GRATING_PITCH'])
    horz_raw = np.array(raw['epicsAll']['MR3K4_pitch'])
    evrs = np.array(raw['timing']['eventcodes'])
    lxt = np.array(raw['epicsAll']['LAS_VIT_TIME'])
    laser = np.array(raw['lightStatus']['laser'])
    events = np.array(raw['timestamp'])

#     try:
#         tt_pos=raw['tt']['fltpos'][:]
#         print('loading TT')
#         tt_posps=raw['tt']['fltpos_ps'][:]
#         tt_posfwhm=raw['tt']['fltposfwhm'][:]
#     except:
#         print('no TT')
#         TT = False

    try:
        print('loading fim0')
        fim0_raw = []
        for i in raw['rix_fim0_raw']:
            fim0_raw.append(np.array((raw['rix_fim0_raw'][i])))
        fim0_raw = np.moveaxis(np.asarray(fim0_raw),0,1)
    except:
        print('no fim0')
        Fim0 = False

    try:
        print('loading fim1')
        fim1_raw = []
        for i in raw['rix_fim1_raw']:
            fim1_raw.append(np.array((raw['rix_fim1_raw'][i])))
        fim1_raw = np.moveaxis(np.asarray(fim1_raw),0,1)
    except:
        print('no fim1')
        Fim1 = False

    try:
        print('loading andor')
        andor_dir_raw = np.array(raw['andor_dir']['full_area'])
    except:
        print('no andor')
        Andor = False

    try:
        print('loading fim2')
        fim2_raw = []
        for i in raw['rix_fim2_raw']:
            fim2_raw.append(np.array((raw['rix_fim2_raw'][i])))
        fim2_raw = np.moveaxis(np.asarray(fim2_raw),0,1)
    except:
        print('no fim2')
        Fim2 = False
        
    RawData.changeValue(xgmd = xgmd,
                       gmd = gmd,
                       energy_raw = energy_raw,
                       mono_encoder = mono_encoder,
                       pitch_raw = pitch_raw,
                       horz_raw = horz_raw,
                       evrs = evrs,
                       lxt = lxt,
                       laser = laser,
                       events = events,
#                        tt_pos = tt_pos,
#                        tt_posps = tt_posps,
#                        tt_posfwhm = tt_posfwhm,
                       fim0_raw = fim0_raw,
                       fim1_raw = fim1_raw,
                       andor_dir_raw = andor_dir_raw,
                       fim2_raw = fim2_raw)
    
    return RawData
    
def load_scans_2_timestamp(small_data_folder,exp,scan):
    
    RawData = RDC_2()
    raw = h5py.File(small_data_folder+'%s_Run%04d.h5' % (exp,scan))
    
    xgmd = np.array(raw['xgmd']['energy'])
    gmd = np.array(raw['gmd']['energy'])
    energy_raw = np.array(raw['epicsAll']['MONO_ENERGY_EV'])
    mono_encoder = np.array(raw['mono_encoder']['value'])
    pitch_raw = np.array(raw['epicsAll']['MONO_GRATING_PITCH'])
    horz_raw = np.array(raw['epicsAll']['MR3K4_pitch'])
    evrs = np.array(raw['timing']['eventcodes'])
    lxt = np.array(raw['epicsAll']['LAS_VIT_TIME'])
    laser = np.array(raw['lightStatus']['laser'])
    events = np.array(raw['timestamp'])

    try:
        tt_pos=raw['tt']['fltpos'][:]
        print('loading TT')
        tt_posps=raw['tt']['fltpos_ps'][:]
        tt_posfwhm=raw['tt']['fltposfwhm'][:]
    except:
        print('no TT')
        TT = False

    try:
        print('loading fim0')
        fim0_raw = []
        for i in raw['rix_fim0_raw']:
            fim0_raw.append(np.array((raw['rix_fim0_raw'][i])))
        fim0_raw = np.moveaxis(np.asarray(fim0_raw),0,1)
    except:
        print('no fim0')
        Fim0 = False

    try:
        print('loading fim1')
        fim1_raw = []
        for i in raw['rix_fim1_raw']:
            fim1_raw.append(np.array((raw['rix_fim1_raw'][i])))
        fim1_raw = np.moveaxis(np.asarray(fim1_raw),0,1)
    except:
        print('no fim1')
        Fim1 = False

    try:
        print('loading andor')
        andor_dir_raw = np.array(raw['andor_dir']['full_area'])
    except:
        print('no andor')
        Andor = False

    try:
        print('loading fim2')
        fim2_raw = []
        for i in raw['rix_fim2_raw']:
            fim2_raw.append(np.array((raw['rix_fim2_raw'][i])))
        fim2_raw = np.moveaxis(np.asarray(fim2_raw),0,1)
    except:
        print('no fim2')
        Fim2 = False
        
    RawData.changeValue(xgmd = xgmd,
                       gmd = gmd,
                       energy_raw = energy_raw,
                       mono_encoder = mono_encoder,
                       pitch_raw = pitch_raw,
                       horz_raw = horz_raw,
                       evrs = evrs,
                       lxt = lxt,
                       laser = laser,
                       tt_pos = tt_pos,
                       tt_posps = tt_posps,
                       tt_posfwhm = tt_posfwhm,
                       fim0_raw = fim0_raw,
                       fim1_raw = fim1_raw,
                       andor_dir_raw = andor_dir_raw,
                       fim2_raw = fim2_raw,
                       events = events)
    
    return RawData
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    