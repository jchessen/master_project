#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:05:42 2023

@author: gaj002
"""

"""This script calculates the dayside reconnection and flux enclosed by the equatorward boundaries of the aurora."""

import os
# import dipole
import matplotlib
import glob
import numpy as np
import fuvpy as fuv
from fuvpy.src.boundaries import boundarymodel_F
from fuvpy.src.plotting import plotboundaries
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
from pysymmetry import visualization
from polplot.grids import equal_area_grid, sdarngrid
import pickle
from pyswipe import SWIPE
import apexpy
from math import atan, pi

from functions import *

#Define grid
dr = 1
M0=5
lowlat=50
grid, mlt_arr = sdarngrid(dlat=1, dlon=2, latmin=50)


#Selects files and selects dates from the datasets
filnavn =['0000_0030','0031_0060','0061_0099','0100_0130','0131_0160']
dates_n = [(2007,8,10),(2010,4,10),(2012,8,10),(2015,4,10),(2017,4,10)]
dates_s = [(2007,12,10),(2010,12,10),(2012, 12, 12),(2014,12,1),(2016,12,10)]
hemi = 'north'
var = 'lbhs' #Which ssusi parameter to perform stat on
c_var_lim = 0.04
Bz_lim = 0


By_val = 'By < -5 nT'
By_titl2 = '-5'


By_val = 'By > 5 nT'
By_titl1 = '5'



tilt_titl2 = -15
tilt_lim2 = 'tilt < -15'

tilt_titl1 = 15
tilt_lim1 = 'tilt > 15'

nans = [np.nan for i in range(0,1030)]




fluxes_pos_By_pos_tilt = []
fluxes_neg_By_pos_tilt = []
fluxes_pos_By_neg_tilt = []
fluxes_neg_By_neg_tilt = []

#Opens files, creates xarrays, estimates boundaries and calculates flux for all files.
for i, fil in enumerate(filnavn):    
    with open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl1}_cvar_{c_var_lim}_{fil}.pickle', 'rb') as handle:
        file_1_info = pickle.load(handle)
    with open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl2}_cvar_{c_var_lim}_{fil}.pickle', 'rb') as handle:
        file_2_info = pickle.load(handle)
    with open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl1}_cvar_{c_var_lim}_{fil}.pickle', 'rb') as handle:
        file_3_info = pickle.load(handle)
    with open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl2}_cvar_{c_var_lim}_{fil}.pickle', 'rb') as handle:
        file_4_info = pickle.load(handle)
    
    file_1 = vaex.open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl1}_cvar_0.04_{fil}.hdf5')
    file_2 = vaex.open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl2}_cvar_0.04_{fil}.hdf5')
    
    file_3 = vaex.open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl1}_cvar_0.04_{fil}.hdf5')
    file_4 = vaex.open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl2}_cvar_0.04_{fil}.hdf5')

    year, month, date = dates_s[i][0], dates_s[i][1],dates_s[i][2],
    year_n, month_n, date_n = dates_n[i][0], dates_n[i][1],dates_n[i][2],
    date_n = [datetime(year_n, month_n, date_n, 9, 36), datetime(year_n, month_n, date_n, 9, 46), datetime(year_n, month_n, date_n, 9, 56)]
    date_s = [datetime(year, month, date, 9, 36), datetime(year, month, date, 9, 46), datetime(year, month, date, 9, 56)]

    try:
        shimg1 = list(file_1.lbhs_pos.values+150)
        shimg2 = list(file_2.lbhs_pos.values+150)
        shimg3 = list(file_3.lbhs_pos_1.values+150)
        shimg4 = list(file_4.lbhs_pos_1.values+150)
    except:pass
    imgs1 = xr.Dataset(
        {
         'shimg': (shimg1)
        },
        coords={
            'mlat': (grid[0]),
            'mlt':(grid[1]),
            'date': date_n
            
            },
    )
    
    imgs2 = xr.Dataset(
        {
         'shimg':  (shimg2)
        },
        coords={
            'mlat': (grid[0]),
            'mlt':(grid[1]),
            'date': date_n
            
            },
    )
    
    imgs3 = xr.Dataset(
        {
         'shimg': (shimg3)
        },
        coords={
            'mlat': (grid[0]),
            'mlt':(grid[1]),
            'date': date_s
            
            },
    )
    
    imgs4 = xr.Dataset(
        {
         'shimg':  (shimg4)
        },
        coords={
            'mlat': (grid[0]),
            'mlt':(grid[1]),
            'date': date_s
            
            },
    )
    #Boundary estimation
    bi1 = detect_boundaries_s_M(imgs1, HiLDA=True)
    bi2 = detect_boundaries_s_M(imgs2, HiLDA=False)
    bi3 = detect_boundaries_w_m(imgs3, HiLDA=False)
    bi4 = detect_boundaries_w_m(imgs4, HiLDA=False)
    bmF1 = boundarymodel_F(bi1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
    bmF2 = boundarymodel_F(bi2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
    bmF3 = boundarymodel_F(bi3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
    bmF4 = boundarymodel_F(bi4,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
    
    #Extracting calculated flux
    flux1 = np.sum(bmF1['dA'][0].values)*(np.pi/180)*1.5
    flux2 = np.sum(bmF2['dA'][0].values)*(np.pi/180)*1.5
    flux3 = np.sum(bmF3['dA'][0].values)*(np.pi/180)*1.5
    flux4 = np.sum(bmF4['dA'][0].values)*(np.pi/180)*1.5
    
    fluxes_pos_By_pos_tilt.append([flux1, file_1_info['avg_Bz'], file_1_info['avg_By'], file_1_info['avg_Vx']])
    fluxes_neg_By_pos_tilt.append([flux2, file_2_info['avg_Bz'], file_2_info['avg_By'], file_2_info['avg_Vx']])
    fluxes_pos_By_neg_tilt.append([flux3, file_3_info['avg_Bz'], file_3_info['avg_By'], file_3_info['avg_Vx']])
    fluxes_neg_By_neg_tilt.append([flux4, file_4_info['avg_Bz'], file_4_info['avg_By'], file_4_info['avg_Vx']])


#Creates figures and plots the flux vs reconnection
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6), sharey=True, sharex=False)
flux1, Bz1, By1, Vx1 = list(zip(*fluxes_pos_By_pos_tilt))
_,_,flux_error1 = bound_uncert(bi1)
_,_,flux_error2 = bound_uncert(bi2)
_,_,flux_error3 = bound_uncert(bi3)
_,_,flux_error4 = bound_uncert(bi4)


ax1.errorbar(rec_func(By1,Bz1,Vx1), flux1, yerr=np.std(flux_error1), fmt="o", color = 'red')
flux2, Bz2, By2, Vx2 = list(zip(*fluxes_neg_By_pos_tilt))

ax1.errorbar(rec_func(By2,Bz2,Vx2), flux2, yerr=np.std(flux_error2), fmt="o", color = 'blue')

for i in range(0,5):
    ax1.text(rec_func(By2,Bz2,Vx2)[i]+0.03, np.add(flux2[i],0.01*1_000_000_000), f'{i+1}')
    ax1.plot([rec_func(By1,Bz1,Vx1)[i],rec_func(By2,Bz2,Vx2)[i]], [flux1[i],flux2[i]], zorder=2, color='black')

ax1.set_title('tilt $>$ 15')
# ax1.legend()
ax1.set_xlabel('Dayside reconnection [kV]')
ax1.set_ylabel('Flux [Wb]')
ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.spines.top.set_visible(False)

flux4, Bz4, By4,Vx4 = zip(*fluxes_neg_By_neg_tilt)
flux3, Bz3, By3, Vx3 = zip(*fluxes_pos_By_neg_tilt)

ax2.errorbar(rec_func(By3,Bz3,Vx3), flux3, yerr=np.std(flux_error3), fmt="o", color = 'red',label='By $>$ 5nT' )
ax2.errorbar(rec_func(By4,Bz4,Vx4), flux4, yerr=np.std(flux_error4), fmt="o", color = 'blue',label='By $<$ -5nT' )

for i in range(0,5):
    ax2.text(rec_func(By3,Bz3,Vx3)[i]+0.03, np.add(flux3[i],0.01*1_000_000_000), f'{i+1}')
    ax2.plot([rec_func(By3,Bz3,Vx3)[i],rec_func(By4,Bz4,Vx4)[i]], [flux3[i],flux4[i]], zorder=2, linestyle='-', color='black', alpha=0.5)

ax1.text(-0.05,1, 'a', transform = ax1.transAxes, size=20)
ax2.text(-0.05,1, 'b', transform = ax2.transAxes, size=20)

ax2.set_title('tilt $<$ -15')
plt.savefig('plots/flux_reconn.pdf', bbox_inches='tight)
ax2.legend(loc='upper left')
ax2.set_xlabel('Dayside reconnection [kV]')





