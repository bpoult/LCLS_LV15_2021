import pandas as pd 

import psana as ps
import numpy as np
import math 
import matplotlib.pyplot as plt
import sys
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter as gf
from sklearn.utils import resample
from filterTools import *

def gauss_norm(x, *p):
    mu, sigma = p
    return np.sqrt(1/(2*np.pi))*(1/sigma)*np.exp(-(((x-mu)**2)/(2*(sigma**2))))

def gauss(x, *p):
    A, mu, sigma, offset = p
    return offset + (A*np.exp(-(((x-mu)**2)/(2*(sigma**2)))))

def gauss_bg(x, *p):
    A, mu, sigma, offset, slope = p
    return offset + slope*x + (A*np.exp(-(((x-mu)**2)/(2*(sigma**2)))))

def myround(x, base=5):
    return base * np.round(x/base)

def centers(x): return (x[:-1]+x[1:])/2

def bin_data_3d(data, x_1, x_2, x_3, edges_1, edges_2, edges_3, ts=None,normalise=None,hits=False):
    centers_1, centers_2, centers_3 = [centers(edges_1),centers(edges_2),centers(edges_3)]
    if ts is None:
        dim_3 = 1
    else:
        if hits:
            dim_3 = len(ts)-1
        else:
            dim_3 = len(ts)
    binned_data = np.zeros((len(centers_1), len(centers_2), len(centers_3), dim_3))
    bin_counts =np.zeros((len(centers_1), len(centers_2), len(centers_3)))
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk_1 = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        for i2, b2 in enumerate(edges_2[0:-1]):
            msk_2 = (x_2>edges_2[i2])&(x_2<=edges_2[i2+1])
            
            for i3, b3 in enumerate(edges_3[0:-1]):
                msk_3 = (x_3>edges_3[i3])&(x_3<=edges_3[i3+1])
                msk = msk_1&msk_2&msk_3
                if hits:
                    try:
                        if normalise is not None:
                            binned_data[i1,i2,i3,:] += np.histogram(np.concatenate((data[msk])),bins = ts,\
                                           weights = np.divide(1,np.repeat(normalise[msk],data[msk].shape[1])))[0]
                        else:
                            binned_data[i1,i2,i3,:] += np.histogram(np.concatenate((data[msk])),bins = ts)[0]
                    except:
                        binned_data[i1,i2,i3,:] += np.zeros(len(ts)-1)
                else:
                    if normalise is not None:
                        binned_data[i1,i2,i3,:] += np.sum(data[msk]/normalise[msk,None], axis = 0)
                    else:
                        binned_data[i1,i2,i3,:] += np.sum(data[msk], axis = 0)
                
                bin_counts[i1,i2,i3]+= data[msk].shape[0]  
    return binned_data, bin_counts

def bin_data_2d(data, x_1, x_2,  edges_1, edges_2,  ts=None,normalise=None,hits=False):
    centers_1, centers_2 = [centers(edges_1),centers(edges_2)]
    if ts is None:
        dim_3 = 1
    else:
        if hits:
            dim_3 = len(ts)-1
        else:
            dim_3 = len(ts)
    binned_data = np.zeros((len(centers_1), len(centers_2), dim_3))
    bin_counts =np.zeros((len(centers_1), len(centers_2)))
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk_1 = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        for i2, b2 in enumerate(edges_2[0:-1]):
            msk_2 = (x_2>edges_2[i2])&(x_2<=edges_2[i2+1])
            msk = msk_1&msk_2
            if hits:
                try:
                    if normalise is not None:
                        binned_data[i1,i2,:] += np.histogram(np.concatenate((data[msk])),bins = ts,\
                                           weights = np.divide(1,np.repeat(normalise[msk],data[msk].shape[1])))[0]
                    else:
                        binned_data[i1,i2,:] += np.histogram(np.concatenate((data[msk])),bins = ts)[0]
                except:
                    binned_data[i1,i2,:] += np.zeros(len(ts)-1)
            else:
                if normalise is not None:
                    binned_data[i1,i2,:] += np.sum(data[msk]/normalise[msk,None], axis = 0)
                else:
                    binned_data[i1,i2,:] += np.sum(data[msk], axis = 0)            
            bin_counts[i1,i2]+= data[msk].shape[0]  
    return binned_data, bin_counts

def bin_norm_data_2d_bootstrap(data,norm, x_1, x_2,  edges_1, edges_2, count_threshold = 1,  ts=None, nsample = None, nrepeat=10,replace = True):
    print('bootstrap bin')
    # bins both data and norm on x_1 and x_2, applies a bin count threshold and calculates the ratio
    centers_1, centers_2 = [centers(edges_1),centers(edges_2)]
    if ts is None:
        dim_3 = 1
    else:
        dim_3 = len(ts)
        
    binned_data = np.zeros((nrepeat,len(centers_1),len(centers_2),dim_3))
    binned_norm = np.zeros((nrepeat,len(centers_1),len(centers_2),dim_3))
    bin_counts =np.zeros((nrepeat,len(centers_1),len(centers_2)))
    
    if nsample is None:
        nsample = len(data)
    
    for i in range(nrepeat):
        
        shots = np.random.choice(np.arange(len(data)), size = nsample, replace = replace)
        for i1,b1 in enumerate(edges_1[0:-1]):

            msk_1 = (x_1[shots]>edges_1[i1])&(x_1[shots]<=edges_1[i1+1])
            for i2, b2 in enumerate(edges_2[0:-1]):
                msk_2 = (x_2[shots]>edges_2[i2])&(x_2[shots]<=edges_2[i2+1])
                msk = msk_1&msk_2

                binned_data[i,i1,i2,:] = np.sum(data[shots][msk])
                binned_norm[i,i1,i2,:] = np.sum(norm[shots][msk])
                bin_counts[i,i1,i2]+= data[shots][msk].shape[0] 
    
    binned_data[bin_counts<count_threshold] = np.nan
    binned_norm[bin_counts<count_threshold] = np.nan
    binned_ratio = np.nanmean(binned_data/binned_norm, axis = 0)
    binned_std = np.nanstd(binned_data/binned_norm, axis = 0)
    
    return binned_ratio, binned_std, bin_counts

def bin_data_2d_bootstrap(data, x_1, x_2,  edges_1, edges_2,  ts=None,normalise=None, nsample = None, nrepeat=500,replace = True):
    print('bootstrap')
    centers_1, centers_2 = [centers(edges_1),centers(edges_2)]
    if ts is None:
        dim_3 = 1
    else:
        if hits:
            dim_3 = len(ts)-1
        else:
            dim_3 = len(ts)
    binned_data = np.zeros((len(centers_1),len(centers_2),dim_3))
    binned_std = np.zeros((len(centers_1),len(centers_2),dim_3))
    bin_counts =np.zeros((len(centers_1),len(centers_2)))
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk_1 = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        for i2, b2 in enumerate(edges_2[0:-1]):
            msk_2 = (x_2>edges_2[i2])&(x_2<=edges_2[i2+1])
            msk = msk_1&msk_2
            
            if normalise is not None:
                if nsample is None:
                    nsample = len(data[msk])
                    
                print(len(data[msk]))
                
                m_tmp, s_tmp = bootstrap_mean(data[msk]/normalise[msk,None], nsample, nrepeat, replace = replace)
                binned_data[i1,i2,:] = np.nanmean(data[msk])
                binned_std[i1,i2,:] = s_tmp
            else:
                if nsample is None:
                    nsample = len(data[msk])
                m_tmp, s_tmp = bootstrap_mean(data[msk], nsample, nrepeat, replace = replace)
                binned_data[i1,i2,:] = np.nanmean(data[msk])
                binned_std[i1,i2,:] = s_tmp
            bin_counts[i1,i2]+= data[msk].shape[0]  
    return binned_data, binned_std, bin_counts

def bin_data_1d(data, x_1, edges_1,  ts=None,normalise=None,hits=False):
    centers_1 = centers(edges_1)
    if ts is None:
        dim_3 = 1
    else:
        if hits:
            dim_3 = len(ts)-1
        else:
            dim_3 = len(ts)
    binned_data = np.zeros((len(centers_1),dim_3))
    bin_counts =np.zeros(len(centers_1))
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        if hits:
            try:
                if normalise is not None:
                    binned_data[i1,:] += np.histogram(np.concatenate((data[msk])),bins = ts,\
                                           weights = np.divide(1,np.repeat(normalise[msk],data[msk].shape[1])))[0]
                else:
                    binned_data[i1,:] += np.histogram(np.concatenate((data[msk])),bins = ts)[0]
            except:
                binned_data[i1,:] += np.zeros(len(ts)-1)
        else:
            if normalise is not None:
                binned_data[i1,:] += np.sum(data[msk]/normalise[msk,None], axis = 0)
            else:
                binned_data[i1,:] += np.sum(data[msk], axis = 0) 
        bin_counts[i1]+= data[msk].shape[0]  
    return binned_data, bin_counts

def bin_data_1d_bootstrap(data, x_1, edges_1,  ts=None,normalise=None, nsample = None, nrepeat=500,replace = True):
    centers_1 = centers(edges_1)
    if ts is None:
        dim_3 = 1
    else:
        if hits:
            dim_3 = len(ts)-1
        else:
            dim_3 = len(ts)
    binned_data = np.zeros((len(centers_1),dim_3))
    binned_std = np.zeros((len(centers_1),dim_3))
    bin_counts =np.zeros(len(centers_1))
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        if normalise is not None:
            if nsample is None:
                nsample = len(data[msk])
            m_tmp, s_tmp = bootstrap_mean(data[msk]/normalise[msk,None], nsample, nrepeat, replace = replace) 
            binned_data[i1,:] = np.nanmean(data[msk])
            binned_std[i1,:] = s_tmp
        else:
            if nsample is None:
                nsample = len(data[msk])
            m_tmp, s_tmp = bootstrap_mean(data[msk], nsample, nrepeat, replace = replace)
            binned_data[i1,:] = np.nanmean(data[msk])
            binned_std[i1,:] = s_tmp
        bin_counts[i1]+= data[msk].shape[0]  
    return binned_data, binned_std, bin_counts

def regress_data_1d(data_x, data_y, x_1, edges_1,normalise=None,coeff_idx = 0, fix_intercept = True, replace = True, nrepeat = 500,plot = False):
    # similar to binning, but performs linear regression on data_x, data_y in each bin and returns result
    centers_1 = centers(edges_1)
    fitted_data = np.zeros(len(centers_1))
    fitted_data_var = np.zeros(len(centers_1))
    fitted_data_err = np.zeros(len(centers_1))
    bin_counts =np.zeros(len(centers_1))
    
    for i1,b1 in enumerate(edges_1[0:-1]):
        
        msk_1 = (x_1>edges_1[i1])&(x_1<=edges_1[i1+1])
        if normalise is not None:
            binned_data[i1,:] = np.sum(data[msk_1]/normalise[msk_1,None], axis = 0)
        else:
            try:
                coeff, var, err = linear_regression(data_x[msk_1],data_y[msk_1],fix_intercept = fix_intercept, replace = replace, nsample = len(data_x))
                fitted_data[i1] = coeff[coeff_idx]   
                fitted_data_var[i1] = var[coeff_idx]
                fitted_data_err[i1] = err[coeff_idx]
                if plot:
                    plt.figure()
                    plt.hist2d(data_x[msk_1],data_y[msk_1],cmap = cmap,bins = 100)
                    plt.plot(data_x[msk_1],straight_line_zero(data_x[msk_1],*coeff))
            except:
                fitted_data[i1] = np.nan   
                fitted_data_var[i1] = np.nan
                fitted_data_err[i1] = np.nan
        bin_counts[i1]+= data_x[msk_1].shape[0]  
    return fitted_data, fitted_data_var, fitted_data_err, bin_counts

def straight_line_zero(x, a):
    return a*x

def straight_line(x, a, b):
    return a*x + b

def linear_regression(x,y, fix_intercept = False, nsample =None, nrepeat = 100, replace = False):
    if nsample is None and replace is True:
        nsample = len(x)
    elif nsample is None and replace is False:
        nsample = int(np.round(len(x)/10)) 
    if fix_intercept:
        coeff, var = bootstrap_fit(straight_line_zero,x,y,[np.average(y)/np.average(x)],nsample,nrepeat,replace = replace)
    else:
        coeff, var = bootstrap_fit(straight_line,x,y,[np.average(y)/np.average(x),0],nsample,nrepeat,replace = replace)
    std_err = var/np.sqrt(nsample)
    return coeff, var, std_err

def invert_linear_regression(coeff, var, err):
    coeff_inv, var_inv, err_inv = [np.zeros(len(coeff)),np.zeros(len(coeff)),np.zeros(len(coeff))]
    if len(coeff)==2:
        coeff_inv[0] = 1/coeff[0]
        coeff_inv[1] = -coeff[1]/coeff[0]
        var_inv[0] = np.abs(coeff_inv[0])*np.abs(var[0]/coeff[0])
        err_inv[0] = np.abs(coeff_inv[0])*np.abs(err[0]/coeff[0])
        
        var_inv[1] = np.abs(coeff_inv[1])*np.sqrt(((var[0]/coeff[0])**2)+((var[1]/coeff[1])**2))
        err_inv[1] = np.abs(coeff_inv[1])*np.sqrt(((err[0]/coeff[0])**2)+((err[1]/coeff[1])**2))
    elif len(coeff)==1:
        coeff_inv[0] = 1/coeff[0]
        var_inv[0] = np.abs(coeff_inv[0])*np.abs(var[0]/coeff[0])
        err_inv[0] = np.abs(coeff_inv[0])*np.abs(err[0]/coeff[0])
    return coeff_inv, var_inv, err_inv

def bootstrap_fit(func, x, y, p0, n_sample, n_repeat, bounds = None, replace = False):
    # similar to curve fit, but the errors in the fit parameters are estimated via bootstrapping
    
    list_coeffs = []
    for i in range(n_repeat):
        shots = np.random.choice(np.arange(len(x)), size = n_sample, replace = replace)
        x_fit, y_fit = x[shots], y[shots]
        try:
            if bounds == None:
                coeff, var = curve_fit(func, x_fit,y_fit, p0 = p0)
            else:
                coeff, var = curve_fit(func, x_fit,y_fit, p0 = p0,bounds = bounds)
            list_coeffs.append(coeff)
        except:
            print('Fit failed :(')

    coeffs = np.array(list_coeffs)
    coeff_mean, coeff_std = np.average(coeffs,axis = 0), np.std(coeffs,axis = 0)
    return coeff_mean, coeff_std

def bootstrap_mean(x, n_sample, n_repeat,axis = 0, bounds = None, replace = False):
    # similar to curve fit, but the errors in the fit parameters are estimated via bootstrapping
    if len(x)<1:
        #print('No data')
        return np.nan, np.nan
    
    means = np.zeros(n_repeat)
    for i in range(n_repeat):
        shots = np.random.choice(np.arange(len(x)), size = n_sample, replace = replace)
        x_m = x[shots]
        means[i] = np.nanmean(x_m)
    mean, mean_std = np.nanmean(means), np.std(means,axis = 0)
    return mean, mean_std


def plot_wav(x,outputs, vlines = None):

    plt.figure(figsize = (4*outputs.shape[0],4))
    for i, output in enumerate(outputs):
        plt.subplot(1,len(outputs),i+1)
        plt.title('Output {:}'.format(i))
        plt.plot(x,output)
        plt.xlabel('bin')
        if vlines is not None:
            for vl in vlines:
                plt.axvline(vl)
    plt.tight_layout()
def hist_wav(outputs, bins = 100,axis = 0):

    plt.figure(figsize = (4*outputs.shape[axis],4))
    for i, output in enumerate(np.rollaxis(outputs,axis)):
        plt.subplot(1,outputs.shape[axis],i+1)
        plt.title('Output {:}'.format(i))
        plt.hist(output,bins = bins)
        plt.xlabel('bin')
    plt.tight_layout()
    
def hist2d_wav(outputs,I0, bins = 100,axis = 0):

    plt.figure(figsize = (4*outputs.shape[axis],4))
    for i, output in enumerate(np.rollaxis(outputs,axis)):
        rho = st.pearsonr(output,I0)[0]
        coeffs = np.polyfit(output,I0,1)
        fit = np.poly1d(coeffs)(output)
        residual = I0-fit
        sigma = np.std(residual)/np.average(I0)
        
        plt.subplot(1,outputs.shape[axis],i+1)
        plt.title('Output {:}, r$^2$ = {:.4f}, sigma = {:.4f}'.format(i,rho,sigma),fontsize = 10)
        plt.hist2d(output,I0,bins = bins)
        plt.plot(output,fit,'r',lw = 0.3)
        plt.xlabel('bin')
    plt.tight_layout()
    

def process_I0_monitor(waveform,threshold=None):
    bg = np.average(waveform[0:50])
    l = -(waveform-bg)
    if threshold is not None:
        l[l<threshold] = 0
    return l


def process_andor(spectra, roi = [0,2048], bg_roi = None):
    px = np.arange(spectra.shape[1])
    msk_tmp = (px>roi[0])&(px<roi[1])
    
    if bg_roi is not None:
        bg_msk = (px>bg_roi[0])&(px<bg_roi[1])
        bgs = np.average(spectra[:,bg_msk],axis = 1)
        specra = spectra - bgs[:,np.newaxis]
        
    return np.sum(andor_tmp[:,msk_tmp],axis = 1)


def fit_andor(spectra, roi = [0,2048]):
    px = np.arange(spectra.shape[1])
    msk_tmp = (px>roi[0])&(px<roi[1])
    spectra = spectra[:,msk_tmp]
    px = px[msk_tmp]
    
    coeffs = np.zeros((spectra.shape[0],4))
    for i, s in enumerate(spectra):
        p0 = [np.max(s),px[np.argmax(s)],20,0]
        coeff, var = curve_fit(gauss, px, s, p0)
        coeffs[i,:] = coeff
        
    return coeffs

def encoder_to_pitch(encoder):
    return (-1.98354918e-05 * encoder) + 6.55128473e-02


def encoder_to_pitch1(encoder, pitch_raw):
    p0 = [1,-7000]
    coeff,var = curve_fit(straight_line,encoder,pitch_raw,p0)
    return (coeff[0] * encoder) + coeff[1]

def mono_energy(GratingPitch, PreMirrorPitch):
    h = 6.62607015E-34
    e_l = 1.602176634E-19
    D = 50000
    c = 299792460
    theta_m1 = 0.03662
    theta_exit = 0.1221413
    pixel_size = 10
    PreMirrorOffset = 90641
    GratingOffset = 63358
    ExitSlitDistance = 20
    order = 1

    pitch_m2 = (PreMirrorPitch-PreMirrorOffset)/1000000
    pitch_g = (GratingPitch - GratingOffset)/1000000
    Alpha = (np.pi/2) - pitch_g + 2*pitch_m2 - theta_m1
    Beta = (np.pi/2) + pitch_g -theta_exit
    EnergyJ = h*c*order*D/(np.sin(Alpha) - np.sin(Beta))
    EnergyeV = EnergyJ/e_l
    
    return EnergyeV

def FIM_shotmask(fimN,method='slice',limits=0.9):
    ''' for FIM, set shot mask.  Methods:
    A.  'range' ,lim=[low,high] (eg. [8e3,3e5])
    B.  'max', lim=maxVal (eg. 8e6)
    C. 'percentile',lim=percentValue (eg. 90)
    D. 'slice', lim=fracValue (eg. 0.9) '''
    
    if method=='range':
        # A. set manually, with range
        RRange = limits
        shtmsk = ((fimN.I_sum>RRange[0]) & (fimN.I_sum<RRange[1]))
        fimN.shot_msk = shtmsk
    
    elif method=='max':
        # B. set manually, max limit only 
        shotmsk = fimN.I_sum<limits
        fimN.shot_msk = shotmsk
    
    elif method=='percentile':
        # C. set manually with percentile
        fimN.set_shot_msk(percentile = limits)

    elif method=='slice':
        ## D. use slice histogram to keep 95% of data
        
        shtmsk = np.full_like(fimN.I_sum,1).astype(bool)
        f0L,f0H,frac,filt_out=slice_histogram(fimN.I_sum,shtmsk,limits,res=100,showplot=True,field='Isum',sub='111')
        fimN.shot_msk = filt_out

