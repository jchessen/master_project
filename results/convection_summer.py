#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:53:43 2023

@author: gaj002
"""

"""This script creates potential plots with overlaid boundaries for both signs of IMF B_y during local summer"""

import os
# import dipole
import matplotlib
import glob
import numpy as np
import fuvpy as fuv
import matplotlib.pyplot as plt
from polplot import pp
from scipy.io import idl
import xarray as xr
import pandas as pd
from datetime import datetime
import vaex
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning) # Turn of all the warnings when np.where contains NaNs.
# from pysymmetry.visualization import polarsubplot
# from pysymmetry import visualization
from polplot import grids
import pickle
from pyswipe import SWIPE
from pyswipe.plot_utils import equal_area_grid, Polarplot, get_h2d_bin_areas
from functions import *

#Define grid
dr = 1
M0=5
lowlat=50
#mlat_, mlt_, mltres_ = grids.equal_area_grid(dr = dr, M0 = M0, N = int((90-lowlat)//dr)) #0.8,8,60
grid, mlt_arr = grids.sdarngrid(dlat=1, dlon=2, latmin=50)

#Opening of files and data selection
filnavn = '0100_0130'
number=4
hemi = 'north'

By_val1 = 'By > 5 nT'
By_titl1 = '5'

By_val2 = 'By < -5 nT'
By_titl2 = '-5'


tilt_titl = 15
tilt_lim = 'tilt > 15'

title = 'Moyal'
nans = [np.nan for i in range(0,1030)]

start, end = 11,15

with open(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl1}_cvar_{c_var_lim}_{filnavn}.pickle', 'rb') as handle:
    file_1_info = pickle.load(handle)
with open(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl2}_cvar_{c_var_lim}_{filnavn}.pickle', 'rb') as handle:
    file_2_info = pickle.load(handle)


file_1 = vaex.open(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl1}_cvar_0.04_{filnavn}.hdf5')
file_2 = vaex.open(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl2}_cvar_0.04_{filnavn}.hdf5')
sfu = vaex.read_csv('data/sfu.csv')


date = [datetime(2023, 8, 28, 9, 36), datetime(2023, 8, 28, 9, 46), datetime(2023, 8, 28, 9, 56)]

if title == 'Moyal':
    try:
        shimg1 = list(file_1.lbhs_pos.values+150)
        shimg2 = list(file_2.lbhs_pos.values+150)
    except:pass

else:
    try:
        shimg1 = list(file_1.lbhs_pos_1.values+150)
        shimg2 = list(file_2.lbhs_pos_1.values+150)
                   
    except:pass


#Calculate mean of SFU for the time period
if file_2_info['avg_tilt'] < 0:
    try:
        sfus = []
        for i in range(start,end):
            if i < 10:
                sfus.append(sfu.mean(sfu.sfu, selection=[(sfu.date > int(f'200{i}1001')) & (sfu.date < int(f'20{i}1231'))]))
            else:
                sfus.append(sfu.mean(sfu.sfu, selection=[(sfu.date > int(f'20{i}1001')) & (sfu.date < int(f'20{i}1231'))]))  
    except:pass
else:
    try:
        sfus = []
        for i in range(start,end):
            if i < 10:
                sfus.append(sfu.mean(sfu.sfu, selection=[sfu.date < int(f'200{i}1001') & sfu.date > int(f'200{i}0401')]))
            else:
                sfus.append(sfu.mean(sfu.sfu, selection=[(sfu.date < int(f'20{i}1001')) & (sfu.date > int(f'20{i}0401'))]))               
    except:pass

imgs1 = xr.Dataset(
    {
     'shimg': (shimg1)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs2 = xr.Dataset(
    {
     'shimg':  (shimg2)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)


#Boundary estimation
bi1 = detect_boundaries_s_M(imgs1, HiLDA=True)
bi2 = detect_boundaries_s_M(imgs2, HiLDA=False)
bmF1 = boundarymodel_F(bi1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF2 = boundarymodel_F(bi2, tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)

#Creation of potential plots
swvel, by, bz = file_1_info['avg_Vx'], file_1_info['avg_By'], file_1_info['avg_Bz']
dptilt, f107 = file_1_info['avg_tilt'], np.nanmean(sfus)
this1 = SWIPE(swvel,by,bz,dptilt,f107,
                             minlat=60)


swvel, by, bz = file_2_info['avg_Vx'], file_2_info['avg_By'], file_2_info['avg_Bz']
dptilt, f107 = file_2_info['avg_tilt'], np.nanmean(sfus)
this2 = SWIPE(swvel,by,bz,dptilt,f107,
                             minlat=60)


plot_potential_(this1,this2, By_val1,By_val2, title,filnavn,bmF1,bmF2,tilt_lim,hemi,tilt_titl,number)


plt.savefig(f'plots/lmfit_tilt_{tilt_titl}_{title}_conv_{filnavn}.pdf')

