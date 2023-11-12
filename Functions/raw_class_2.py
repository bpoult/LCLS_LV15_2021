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

font = {'size'   : 16}
mpl.rc('font', **font)

cmap = plt.cm.get_cmap('terrain').reversed()

class RawData_2:
    _defaults = ['xgmd',
                'gmd',
                'energy_raw',
                'mono_encoder',
                'pitch_raw',
                'horz_raw',
                'evrs',
                'lxt',
                'laser'
                'event']

    _default_value = None
    
    def __init__(self, **kwargs):
        self.__dict__.update(dict.fromkeys(self._defaults, self._default_value))
        self.__dict__.update(kwargs)

    def changeValue(self, **kwargs):
        self.__dict__.update(kwargs)

    def getKeys(self):
        return self.__dict__.keys()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    