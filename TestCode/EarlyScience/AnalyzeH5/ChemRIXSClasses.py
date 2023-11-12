''' chemRIXS classes for analysis.  Prepared by Dougie, from h5Online.ipynb'''

import pandas as pd 

import psana as ps
import numpy as np
import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import h5py
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter as gf
from sklearn.utils import resample
sys.path.append('/reg/data/ana16/rix/rixlv1519/results/LCLS_LV15_2021/TestCode/EarlyScience/AnalyzeH5/')
from chemRIXSAnalysis import *

import matplotlib
font = {'size'   : 16}

matplotlib.rc('font', **font)
cmap = plt.cm.get_cmap('terrain').reversed()



class I_monitor():
    def __init__(self):
        '''intensity sums for each shot (sum over the waveform for the different measures/channels and add to I_sum)
        axis 0 is the shot number, axis 1 is the different measures '''
        self.I_sums = None
        # [slope, offset]
        self.calibration = np.array([1,0])
        
    def load_run(self,data):
        if self.I_sums == None:
            self.I_sums = self.process(data)
        else:
            self.I_sums = np.concatenate((self.I_sums, self.process(data)))

    def process(self, data):
        self.I_sums = np.sum(data,axis = data.shape[-1])
        
class fim(I_monitor):   
    def __init__(self):
        self.I_sums = None
        self.I_sum = None
        self.average_waveforms = None
        self.calibration = np.array([1.0,0.0])
        self.calibrations = None
        self.calibration_error = np.array([0.0,0.0])
        self.threshold = -5000
        self.ts = np.arange(256) #ts= shot number (event time)
        self.msk = (self.ts>0)&(self.ts<257)
        self.bg_msk = (self.ts>0)&(self.ts<50)
        self.shot_msk = None
        self.detector_msk = None
        self.nrepeat = 1000
        
    def load_run(self,data):
        if self.I_sums is None:
            self.I_sums, self.average_waveforms = self.process(data)
            self.ts = np.arange(data.shape[-1])
            self.set_detector_msk(np.arange(self.I_sums.shape[1]))
            self.calibrations = np.tile(np.array([1.0,0.0]),(self.I_sums.shape[1],1))
            self.average_I()
            self.shot_msk = np.ones(self.I_sum.shape[0]).astype('bool')
        else:
            I_tmp, wfs_tmp = self.process(data)
            self.I_sums = np.concatenate((self.I_sums, I_tmp))
            self.average_waveforms = (self.average_waveforms + wfs_tmp)/2
            self.average_I()
            self.shot_msk = np.ones(self.I_sum.shape[0]).astype('bool')
            
    def average_I(self):
        ''' intensity averaged over selected channels'''
        self.I_sum = self.calibration[0]*np.average(self.I_sums[:,self.detector_msk],axis = 1) + self.calibration[1]
        
    def process(self,data):
        ''' returns I_sums (sum over all channels), average_waveform '''
        tmp = np.zeros((data.shape[0],data.shape[1])) #shots x channels
        av_wfs = np.zeros((data.shape[1],data.shape[2])) #channel x time 
        for i, wfs in enumerate(np.rollaxis(data,1)):
            #for each channel, take the average waveform of all shots, after subtracting background
            bgs = np.average(wfs[:,self.bg_msk],axis = 1)
            ls = -(wfs-bgs[:,np.newaxis])
            ls[ls<self.threshold] = 0
            tmp[:,i] = np.sum(ls,axis = 1) #Isum after background subtraction for each shot (sum over waveform time)
            av_wfs[i,:] = np.average(ls,axis = 0)
        return tmp, av_wfs
    
    
    def calibrate(self, It,mask_shots = True):
        if mask_shots:
            coeff, var, err = linear_regression(It[self.shot_msk],self.I_sum[self.shot_msk], nrepeat = self.nrepeat, nsample = len(self.I_sum), replace = True, fix_intercept = False)
        else:
            coeff, var, err = linear_regression(It,self.I_sum, nrepeat = self.nrepeat, nsample = len(self.I_sum), replace = True, fix_intercept = False)
        coeff, var, err = invert_linear_regression(coeff, var, err)
        self.calibration = coeff
        self.calibration_err = err
        self.average_I()
        
    def calibrate_all(self, It,mask_shots = True):
        for i, d_sums in enumerate(np.rollaxis(self.I_sums[:,self.detector_msk],axis = 1)):
            if mask_shots:
                coeff, var, err = linear_regression(It[self.shot_msk],d_sums[self.shot_msk], nrepeat = self.nrepeat, nsample = len(d_sums), replace = True, fix_intercept = False)
            else:
                coeff, var, err = linear_regression(It,d_sums, nrepeat = self.nrepeat, nsample = len(d_sums), replace = True, fix_intercept = False)
            coeff, var, err = invert_linear_regression(coeff, var, err)
            self.I_sums[:,self.indicies[i]] = coeff[0]*self.I_sums[:,self.indicies[i]]+ coeff[1]
            self.calibrations[i,:] = coeff
        self.calibration = np.array([1.0,0.0])
        self.average_I()
    
    def set_detector_msk(self, indicies):
        '''set which channels to look at'''
        self.indicies = np.array(indicies)
        self.detector_msk = np.zeros(8).astype('bool')
        self.detector_msk[self.indicies] = True
        self.average_I()
    
    def plot_waveforms(self, msked = False,figsize = None):
        if figsize == None: 
            figsize = (4*self.average_waveforms.shape[0],8)
        if not msked:
            plt.figure(figsize = figsize)
            for i, wf in enumerate(self.average_waveforms):
                plt.subplot(2,int(len(self.average_waveforms)/2),i+1)
                plt.title('Detector {:}'.format(i))
                plt.plot(self.ts,wf)
                plt.xlabel('t')
                plt.axvline(self.ts[self.msk][0], color = 'r', lw = 0.5,label='msk')
                plt.axvline(self.ts[self.msk][-1], color = 'r', lw = 0.5)
                
                plt.axvline(self.ts[self.bg_msk][0], color = 'g', lw = 0.5,label='bg_msk')
                plt.axvline(self.ts[self.bg_msk][-1], color = 'g', lw = 0.5)
                plt.legend()
        else:
            #only plot channels specified in detector_msk
            plt.figure(figsize = figsize)
            for i, wf in enumerate(self.average_waveforms[self.detector_msk,:]):
                plt.subplot(1,len(self.average_waveforms[self.detector_msk,:]),i+1)
                plt.title('Detector {:}'.format(self.indicies[i]))
                plt.plot(self.ts,wf)
                plt.xlabel('t')
                plt.axvline(self.ts[self.msk][0], color = 'r', lw = 0.5,label='msk')
                plt.axvline(self.ts[self.msk][-1], color = 'r', lw = 0.5)
                
                plt.axvline(self.ts[self.bg_msk][0], color = 'g', lw = 0.5,label='bg_msk')
                plt.axvline(self.ts[self.bg_msk][-1], color = 'g', lw = 0.5)
                plt.legend()
        plt.tight_layout()
                
    def hist_all(self, msked = False, bins = 100,figsize = None):
        if figsize == None: 
            figsize = (5*self.average_waveforms.shape[0],4)
        if not msked:
            plt.figure(figsize = figsize)
            for i, dat in enumerate(np.rollaxis(self.I_sums,1)):
                plt.subplot(2,int(len(self.average_waveforms)/2),i+1)
                plt.title('Detector {:}'.format(i))
                plt.hist(dat, bins = bins)
                plt.xlabel('sum')
        else:
            plt.figure(figsize = figsize)
            for i, dat in enumerate(np.rollaxis(self.I_sums[:,self.detector_msk],1)):
                plt.subplot(1,len(self.average_waveforms[self.detector_msk,:]),i+1)
                plt.title('Detector {:}'.format(self.indicies[i]))
                plt.hist(dat, bins = bins)
                plt.xlabel('sum')
        plt.tight_layout()
                
    def hist2d_all(self, It, msked = False, bins = 100, fs = 10, scatter = False,figsize = None, zoom = False, label = True):
        ''' 2d histogram of given fim and It'''
        if not msked:
            if figsize == None: 
                figsize = (5*self.average_waveforms.shape[0],4)
            plt.figure(figsize = figsize)
            for i, dat in enumerate(np.rollaxis(self.I_sums,1)):
                coeff, var, err = linear_regression(It,dat)
                coeff, var, err = invert_linear_regression(coeff, var, err)
                r = st.pearsonr(dat,It)[0]
                std = np.std(It[self.shot_msk]-straight_line(dat[self.shot_msk],coeff[0],coeff[1]))/np.average(It)
                plt.subplot(2,int(len(self.average_waveforms)/2),i+1)
                plt.title('Detector {:}, $r^2$ = {:.4f}, $\sigma$ = {:.4f} \%'.format(i,r,std*100), fontsize = fs)
                if scatter:
                    plt.scatter(dat,It, s = 2)
                    plt.plot(dat,straight_line(dat,coeff[0],coeff[1]), color = 'k',lw =0.5,\
                                           label = 'y = ({:.4f} $\pm$ {:.4f})x + ({:.0f} $\pm$ {:.0f})'.format(coeff[0],var[0],coeff[1],var[1]))
                else:
                    plt.hist2d(dat,It, bins = bins,cmap = cmap)
                plt.plot(dat,straight_line(dat,coeff[0],coeff[1]), color = 'k',lw =0.5,\
                                           label = 'y = ({:.4f} $\pm$ {:.4f})x + ({:.0f} $\pm$ {:.0f})'.format(coeff[0],var[0],coeff[1],var[1]))   
                plt.xlabel('I0')
                plt.ylabel('It')
                plt.axvline(np.min(dat[self.shot_msk]),color = 'r',lw=0.5,label='shot_msk')
                plt.axvline(np.max(dat[self.shot_msk]),color = 'r',lw=0.5)
                if label:
                    plt.legend(fontsize = fs)
                
        else:
            if figsize == None: 
                figsize = (4*self.average_waveforms[self.detector_msk,:].shape[0],4)
            plt.figure(figsize = figsize)
            
            for i, dat in enumerate(np.rollaxis(self.I_sums[:,self.detector_msk],1)):
                print(i)
                coeff, var, err = linear_regression(It,dat)
                coeff, var, err = invert_linear_regression(coeff, var, err)
                r = st.pearsonr(dat,It)[0]
                std = np.std(It[self.shot_msk]-straight_line(dat[self.shot_msk],coeff[0],coeff[1]))/np.average(It)
                plt.subplot(1,len(self.average_waveforms[self.detector_msk,:]),i+1)
                plt.title('Detector {:}, $r^2$ = {:.4f}, $\sigma$ = {:.4f} \%'.format(self.indicies[i],r,std*100), fontsize = fs)
                if scatter:
                    plt.scatter(dat,It, s = 2)
                    plt.plot(dat,straight_line(dat,coeff[0],coeff[1]), color = 'k',lw =0.5,\
                                           label = 'y = ({:.4f} $\pm$ {:.4f})x + ({:.0f} $\pm$ {:.0f})'.format(coeff[0],var[0],coeff[1],var[1]))
                else:
                    plt.hist2d(dat,It, bins = bins,cmap = cmap)
                    plt.plot(dat,straight_line(dat,coeff[0],coeff[1]), color = 'k',lw =0.5,\
                                           label = 'y = ({:.4f} $\pm$ {:.4f})x + ({:.0f} $\pm$ {:.0f})'.format(coeff[0],var[0],coeff[1],var[1]))
                plt.xlabel('I0')
                plt.ylabel('It')
                plt.axvline(np.min(dat[self.shot_msk]),color = 'r',lw=0.5,label='shot_msk')
                plt.axvline(np.max(dat[self.shot_msk]),color = 'r',lw=0.5)
                if zoom == True:
                    plt.xlim(np.min(dat[self.shot_msk]),np.max(dat[self.shot_msk]))
                    plt.ylim(coeff[0]*np.min(dat[self.shot_msk])+coeff[1],coeff[0]*np.max(dat[self.shot_msk])+coeff[1])
                if label:
                    plt.legend(fontsize = fs)
        plt.tight_layout()
                
    def hist(self, bins = 100,figsize = None):
        ''' histogram of I_sum for given channels'''
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        plt.hist(self.I_sum, bins = bins)
        plt.title('Detectors: {:}'.format(self.indicies))
        plt.xlabel('Sum')
        plt.axvline(np.min(self.I_sum[self.shot_msk]),color = 'r',lw=0.5,label='shot_msk')
        plt.axvline(np.max(self.I_sum[self.shot_msk]),color = 'r',lw=0.5)
        plt.legend()
        
    def hist2d(self,It, bins = 100, fs = 10,figsize = None):
        ''' 2d histogram of given self vs It.  plot linear regression/correlation.
        fs=font size '''
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        coeff, var, err = linear_regression(It,self.I_sum)
        coeff, var, err = invert_linear_regression(coeff, var, err)
        r = st.pearsonr(self.I_sum,It)[0]
        std = np.std(It[self.shot_msk]-straight_line(self.I_sum[self.shot_msk],coeff[0],coeff[1]))/np.average(It)
        plt.hist2d(self.I_sum,It, bins = bins,cmap = cmap)
        plt.colorbar()
        plt.title('Detectors = {:}, $r^2$ = {:.4f}, $\sigma$ = {:.4f} \%'.format(self.indicies,r,std*100), fontsize = fs)
        plt.plot(self.I_sum,straight_line(self.I_sum,coeff[0],coeff[1]), color = 'k',lw =0.5,\
                                           label = 'y = ({:.4f} $\pm$ {:.4f})x + ({:.0f} $\pm$ {:.0f})'.format(coeff[0],var[0],coeff[1],var[1]))
        plt.legend(fontsize = fs)                                      
        plt.xlabel('I0')
        plt.ylabel('It')
        plt.axvline(np.min(self.I_sum[self.shot_msk]),color = 'r',lw=0.5)
        plt.axvline(np.max(self.I_sum[self.shot_msk]),color = 'r',lw=0.5)
                    
        plt.tight_layout()
        
    def plot_waveform(self):
        ''' plot average detector response''' 
        plt.figure()
        plt.plot(np.average(self.average_waveforms[self.detector_msk,:],axis = 0))
        plt.title('FIM Detectors: {:}'.format(self.indicies))
        plt.xlabel('ts')

        plt.tight_layout()
    def set_msk(self, roi):
        '''only look at ts w/in limits of roi'''
        self.msk = (self.ts>roi[0])&(self.ts<roi[1])
    def set_bg_msk(self, roi):
        self.bg_msk = (self.ts>roi[0])&(self.ts<roi[1])
    def set_shot_msk(self,I_roi = None, percentile = None):
        '''filter shots by I_sum limits or or select values within some percentile of I_sum'''
        if I_roi is not None:
            self.shot_msk = (self.I_sum>I_roi[0])&(self.I_sum<I_roi[1])
        if percentile is not None:
            p_mx = np.percentile(self.I_sum,percentile)
            p_mn = np.percentile(self.I_sum,100-percentile)
            self.shot_msk = (self.I_sum>p_mn)&(self.I_sum<p_mx)

        
class andor(I_monitor): 
    def __init__(self):
        self.I_sums = None
        self.coeffs = None
        self.I_sum = None
        self.I_fit = None
        self.average_waveform = None
        self.calibration = np.array([1,0])
        self.threshold = -500
        self.px = np.arange(2048)
        self.msk = (self.px>0)&(self.px<2050)
        self.fit_msk = (self.px>0)&(self.px<2050)
        self.bg_msk = (self.px>0)&(self.px<500)
        
    def load_run(self,data,fit = False):
        if self.I_sums is None:
            self.px = np.arange(data.shape[1])
            self.I_sums, self.average_waveform, self.coeffs = self.process(data, fit = fit) 
            self.average_I()
        else:
            I_tmp, wfs_tmp, c_tmp = self.process(data, fit = fit)
            self.I_sums = np.concatenate((self.I_sums, I_tmp))
            self.average_waveform = (self.average_waveform + wfs_tmp)/2
            if self.coeffs is not None:
                self.coeffs = np.concatenate((self.coeffs,c_tmp),axis = 0)
            self.average_I()
            
    def process(self, data, fit = False):
        bgs = np.average(data[:,self.bg_msk],axis = 1)
        data = data - bgs[:,np.newaxis]
        data[data<self.threshold]=0
        av = np.average(data,axis = 0)
        sums = np.sum(data[:,self.msk],axis = 1)
        coeffs = None
        if fit: 
            data = data[:,self.fit_msk]
            px = self.px[self.fit_msk]
            coeffs = np.zeros((data.shape[0],4))
            for i, s in enumerate(data):
                p0 = [np.max(s),px[np.argmax(s)],20,0]
                coeff, var = curve_fit(gauss, px, s, p0)
                coeffs[i,:] = coeff
        
        return sums, av, coeffs
    
    def average_I(self):
        self.I_sum = self.calibration[0]*self.I_sums + self.calibration[1]
        if self.coeffs is not None:
            tmp = np.multiply(self.coeffs[:,0], self.coeffs[:,2])
            self.I_fit = self.calibration[0]*tmp + self.calibration[1]
        else: 
            self.I_fit = None
    def plot_waveform(self,figsize = None):
        ''' plot average waveform and masks'''
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        plt.suptitle('andor')
        plt.plot(self.px,self.average_waveform)
        plt.xlabel('px')
        plt.axvline(self.px[self.msk][0], color = 'r', lw = 0.5,label='msk')
        plt.axvline(self.px[self.msk][-1], color = 'r', lw = 0.5)
                
        plt.axvline(self.px[self.bg_msk][0], color = 'g', lw = 0.5,label='bg_msk')
        plt.axvline(self.px[self.bg_msk][-1], color = 'g', lw = 0.5)
        
        plt.axvline(self.px[self.fit_msk][0], color = 'b', lw = 0.5,label='fit_msk')
        plt.axvline(self.px[self.fit_msk][-1], color = 'b', lw = 0.5)
        plt.legend()
        
    def hist_coeffs(self):
        if self.coeffs is None:
            print('Not fitted')
        else:
            plt.figure(figsize = (4*5,4))
            names = ['Amp', 'mu', 'sigma', 'offset']
            for i, c in enumerate(np.rollaxis(self.coeffs,1)):
                plt.subplot(1,5,i+1)
                plt.hist(c)
                plt.xlabel(names[i])
        plt.tight_layout()
            
        
    def hist(self, bins = 100,figsize = None):
        if figsize == None: 
            figsize = (7,5)
        
        if self.I_fit is None:
            plt.figure(figsize = figsize)
            plt.hist(self.I_sum,bins = bins)
            plt.xlabel('Sum')
        else:
            if figsize == None: 
                figsize = (7,5)
            plt.figure(figsize = figsize)
            plt.subplot(1,2,1)
            plt.hist(self.I_sum,bins = bins)
            plt.xlabel('Sum')
            plt.title('Sum')
            plt.subplot(1,2,2)
            plt.hist(self.I_fit,bins = bins)
            plt.xlabel('Fit AUC')
            plt.title('AUC')
    
    def set_msk(self, roi):
        ''' roi of area that is summed for Isum''' 
        self.msk = (self.px>roi[0])&(self.px<roi[1])
    def set_bg_msk(self, roi):
        self.bg_msk = (self.px>roi[0])&(self.px<roi[1])
    def set_fit_msk(self, roi):
        ''' roi for gaussian fit of andor data''' 
        self.fit_msk = (self.px>roi[0])&(self.px<roi[1])

class mono_spectrum(): 
    def __init__(self,grating_pos, I0, It,msk):
        self.pitches = grating_pos
        self.I0 = I0
        self.It = It
        self.I0_binned = None
        self.It_binned = None
        self.I_bin_count = None
        self.transmission_ss = It/I0 #single shot transmission
        self.msk = msk
        self.bin_msk=None
        self.transmission = None
        self.transmission_err = None
        self.bin_count = None
        # NoI0
        self.pitch_bins = None
        self.pitch_bin_centers = None
        self.pitch_bin_means = None
        
        self.nrepeat = 20 
        self.nsample = len(I0)
        
    def hist_I0(self,bins = 100,figsize = (6,5)):
        plt.figure(figsize = figsize)
        plt.hist(self.I0[self.msk],bins = bins)
        plt.xlabel('I0')
        
    def set_pitch_bins(self,pitch_bins):
        self.pitch_bins = pitch_bins
        self.pitch_bin_centers = centers(self.pitch_bins) #same as pitch_bins o.O
        
    def plot_bins(self,figsize = None):
        ''' plot unique pitch values and pitch value per shot so as to determine what pitch binning used'''
        if figsize == None: 
            figsize = (10,5)
        plt.subplot(1,2,1)
        plt.plot(np.unique(self.pitches),label='unique pitches')
        plt.ylabel('Grating Pitch')
        for b in self.pitch_bins:
            plt.axhline(b,color = 'C1',lw = 0.5)
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.pitches,label='pitches')
        plt.xlabel('shot number')
        plt.ylabel('Grating Pitch')
        plt.legend()
        for b in self.pitch_bins:
            plt.axhline(b,color = 'C1',lw = 0.5) 
        plt.tight_layout()

    def calc_spectrum_mean(self,nrepeat = None, nsample = None,replace = True):
        '''calc absorption spectrum , binned wrt encoder value. Normalize shot by shot, then bin.
        estimate error w/ bootstrapping (process subsetss of data, determine deviations): 
        nrepeat=number of trials to do with a sample of size = nsample.
        replace = whether or not to replace subset of data before next trial.'''
        if nrepeat is not None:
            self.nrepeat = nrepeat
        else:
            nrepeat = self.nrepeat
        if nsample is None:
            nsample = len(self.I0[self.msk])
            
        t_binned, t_binned_std, t_bc = bin_data_1d_bootstrap(self.transmission_ss[self.msk], self.pitches[self.msk],self.pitch_bins,\
                                                            nrepeat = nrepeat, nsample = nsample)
        t_binned,  t_bc = bin_data_1d(self.transmission_ss[self.msk], self.pitches[self.msk],self.pitch_bins)
        self.pitch_bin_means, bc = bin_data_1d(self.pitches[self.msk], self.pitches[self.msk],self.pitch_bins)
        self.pitch_bin_means = self.pitch_bin_means.flatten()
        self.bin_msk=bc>0.5
        self.transmission = t_binned.flatten()/t_bc.flatten()
        self.transmission_err = t_binned_std.flatten()/t_bc.flatten()
        self.bin_count = t_bc.flatten()
        
        self.dt = (self.transmission-1)*100
        self.dt_err = self.transmission_err*100
        
        self.OD = -np.log(self.transmission)*1000
        self.OD_err = np.abs(self.transmission_err/self.transmission)*1000

    def calc_spectrum_NoI0(self,nrepeat = None, nsample = None,replace = True):
        '''calc absorption spectrum , binned wrt encoder value. Normalize shot by shot, then bin.
        estimate error w/ bootstrapping (process subsetss of data, determine deviations): 
        nrepeat=number of trials to do with a sample of size = nsample.
        replace = whether or not to replace subset of data before next trial.'''
        if nrepeat is not None:
            self.nrepeat = nrepeat
        else:
            nrepeat = self.nrepeat
        if nsample is None:
            nsample = len(self.I0[self.msk])
            
        t_binnedNoI0, t_binned_stdNoI0, t_bcNoI0 = bin_data_1d_bootstrap(self.It[self.msk], self.pitches[self.msk],self.pitch_bins,\
                                                            nrepeat = nrepeat, nsample = nsample)
        t_binnedNoI0,  t_bcNoI0 = bin_data_1d(self.It[self.msk], self.pitches[self.msk],self.pitch_bins)
        self.pitch_bin_meansNoI0, bc = bin_data_1d(self.pitches[self.msk], self.pitches[self.msk],self.pitch_bins)
        self.pitch_bin_meansNoI0 = self.pitch_bin_meansNoI0.flatten()
        self.bin_msk=bc>0.5
        self.transmissionNoI0 = t_binnedNoI0.flatten()/t_bcNoI0.flatten()
        self.transmission_errNoI0 = t_binned_stdNoI0.flatten()/t_bcNoI0.flatten()
        self.bin_count = t_bcNoI0.flatten()
        
        self.dtNoI0 = (self.transmissionNoI0-1)*100
        self.dt_errNoI0 = self.transmission_errNoI0*100
        
        self.ODNoI0 = -np.log(self.transmissionNoI0)*1000
        self.OD_errNoI0 = np.abs(self.transmission_errNoI0/self.transmissionNoI0)*1000

    def calc_spectrum_mean_binned(self,I0_bins, count_threshold = 1, bootstrap = False,nrepeat = None, nsample = None,replace = True):
        ''' Bins on I0, encoder.  Takes ration of It/I0, then sum over I0 to get spectrum. '''
        if bootstrap:
            if nrepeat is not None:
                self.nrepeat = nrepeat
            if nsample is None:
                nsample = len(self.I0[self.msk])
            self.I0_binned, I0_bin_count = bin_data_2d(self.I0[self.msk], self.I0[self.msk],self.pitches[self.msk],I0_bins, self.pitch_bins)
            self.It_binned, It_bin_count = bin_data_2d(self.It[self.msk], self.I0[self.msk],self.pitches[self.msk],I0_bins, self.pitch_bins)
            print(nsample, nrepeat, replace)
            bm_tmp, bs, bc_tmp = bin_norm_data_2d_bootstrap(self.It[self.msk], self.I0[self.msk], self.I0[self.msk], self.pitches[self.msk],\
                                                            I0_bins,self.pitch_bins,nsample = nsample,replace = replace, count_threshold = count_threshold)
            np.nanmean(bs,axis = 0)
            self.transmission_binned_err = np.nanmean(bs,axis = 0).flatten()
        else:
            self.I0_binned, I0_bin_count = bin_data_2d(self.I0[self.msk], self.I0[self.msk],self.pitches[self.msk],I0_bins, self.pitch_bins)
            self.It_binned, It_bin_count = bin_data_2d(self.It[self.msk], self.I0[self.msk],self.pitches[self.msk],I0_bins, self.pitch_bins)
            self.transmission_binned_err = np.zeros(self.I0_binned.shape[1]).flatten()
            #self.I0_binned_std, self.It_binned_std = np.zeros(self.I0_binned.shape), np.zeros(self.I0_binned.shape)
        
        
        self.pitch_bin_means, bc_p = bin_data_1d(self.pitches[self.msk], self.pitches[self.msk],self.pitch_bins)
        
        count_msk = (self.I0_binned<count_threshold)&(self.It_binned<count_threshold)
        self.I0_binned[count_msk] = np.nan
        self.It_binned[count_msk] = np.nan
        self.I0_bins = I0_bins 
        self.count_threshold = count_threshold
        self.transmission_binned_standard_err = self.transmission_binned_err/np.sqrt(nsample)
        
        self.transmission_binned = np.nanmean(self.It_binned/self.I0_binned,axis = 0).flatten()
        self.dt_binned = (self.transmission_binned-1)*100
        self.dt_binned_err = self.transmission_binned_err*100
        
        #ratio_err = np.sqrt((np.power(self.I0_binned_std/self.I0_binned,2))+\
        #                                      (np.power(self.It_binned_std/self.It_binned,2)))
        #ratio = self.I0_binned[count_msk]/self.It_binned[count_msk]
        #self.transmission_binned_err = np.sqrt(np.nanmean(np.power(ratio_err,2),axis = 0))
        self.OD_binned = -np.log(self.transmission_binned)*1000
        self.OD_binned_err = np.abs(self.transmission_binned_err/self.transmission_binned)*1000
        self.OD_binned_standard_err = np.abs(self.transmission_binned_standard_err/self.transmission_binned)*1000
        
        
    def bin_I(self,I0_bins):
        self.I0_binned, self.I_bin_count = bin_data_1d(self.I0[self.msk], self.I0[self.msk],I0_bins)
        self.It_binned, tmp = bin_data_1d(self.It[self.msk], self.I0[self.msk],I0_bins)
        
    def plot_dt(self,figsize = None, binned = True):
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        plt.figure()
        if binned:
            plt.plot(self.pitch_bin_centers, self.dt_binned)
            plt.fill_between(self.pitch_bin_centers, self.dt_binned-self.dt_binned_err,self.dt_binned+self.dt_binned_err, alpha = 0.3, color = 'C1')
        else:
            plt.plot(self.pitch_bin_centers, self.dt)
            plt.fill_between(self.pitch_bin_centers, self.dt-self.dt_err,self.dt+self.dt_err, alpha = 0.3, color = 'C1')
        plt.xlabel('Mono Encoder')
        plt.ylabel('dt (\%)')
        
    def plot_OD(self,figsize = None,binned = True):
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        plt.figure()
        if binned:
            plt.plot(self.pitch_bin_centers, self.OD_binned)
            plt.fill_between(self.pitch_bin_centers, self.OD_binned-self.OD_binned_err,self.OD_binned+self.OD_binned_err, alpha = 0.3, color = 'C1')
        else:
            plt.plot(self.pitch_bin_centers, self.OD)
            plt.fill_between(self.pitch_bin_centers, self.OD-self.OD_err,self.OD+self.OD_err, alpha = 0.3, color = 'C1')
        plt.xlabel('Mono Encoder')
        plt.ylabel('Absorption (mOD)')
        
    def plot_yield(self,figsize = None):
        if figsize == None: 
            figsize = (7,5)
        plt.figure(figsize = figsize)
        plt.figure()
        plt.plot(self.pitch_bin_centers, self.transmission)
        plt.fill_between(self.pitch_bin_means, self.transmission-self.transmission_err,self.transmission+self.transmission_err, alpha = 0.3, color = 'C1')
        plt.xlabel('Mono Encoder')
        plt.ylabel('Fluoresence yield/I0')
    
    
    # Rebin data, calculate transmission in different ways, calibration
        