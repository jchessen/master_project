#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:31:29 2023

@author: gaj002
"""

"""This script creates plots of the Moyal mean, total mean, and the difference between them during local summer
    in the Southern Hemisphere for all the datasets. """



import os
import matplotlib
import glob
import numpy as np
import fuvpy as fuv
from fuvpy import boundarymodel_F
import matplotlib.pyplot as plt
from polplot import pp
from scipy.io import idl
import xarray as xr
import pandas as pd
from datetime import datetime
import vaex
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning) # Turn of all the warnings when np.where contains NaNs.
from polplot import grids
import pickle
from pyswipe import SWIPE
from functions import *


#Define grid
dr = 1
M0=5
lowlat=50
grid, mlt_arr = grids.sdarngrid(dlat=1, dlon=2, latmin=50)

#Determines which files to laod
filnavn1 = '0000_0030'
filnavn2 = '0031_0060'
filnavn3 = '0061_0099'


#Data selection based on hemisphere, IMF By and dipole tilt angle
hemi = 'south'

#Select one of the signs of IMF B_y at a time
By_val = 'By < -5 nT'
By_titl = '-5'


By_val = 'By > 5 nT'
By_titl = '5'


tilt_titl = -15
tilt_lim = 'tilt < -15'

#Checks if the HiLDA-spot is to be taken into account by the boundary estimation
if (By_titl == '5') and (tilt_titl > 0):
    HiLDA = True
else: HiLDA = False

nans = [np.nan for i in range(0,1030)]
date = [datetime(2023, 8, 28, 9, 36), datetime(2023, 8, 28, 9, 46), datetime(2023, 8, 28, 9, 56)]

#Open files
file1 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn1}.hdf5')
file2 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn2}.hdf5')
file3 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn3}.hdf5')

with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_{c_var_lim}_{filnavn1}.pickle', 'rb') as handle:
    file_info1 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_{c_var_lim}_{filnavn2}.pickle', 'rb') as handle:
    file_info2 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_{c_var_lim}_{filnavn3}.pickle', 'rb') as handle:
    file_info3 = pickle.load(handle)
    
    
#Loads intensities from files, adds offset, and creates xarrays
try:
    shimg_moyal1 = list(file1.lbhs_pos.values+150)
    shimg_mean1 = list(file1.lbhs_pos_1.values+150)
    shimg_moyal2 = list(file2.lbhs_pos.values+150)
    shimg_mean2 = list(file2.lbhs_pos_1.values+150)
    shimg_moyal3 = list(file3.lbhs_pos.values+150)
    shimg_mean3 = list(file3.lbhs_pos_1.values+150)

except:pass
imgs_moyal1 = xr.Dataset(
    {
     'shimg': (shimg_moyal1)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs_mean1 = xr.Dataset(
    {
     'shimg':  (shimg_mean1)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)
imgs_moyal2 = xr.Dataset(
    {
     'shimg': (shimg_moyal2)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs_mean2 = xr.Dataset(
    {
     'shimg':  (shimg_mean2)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)
imgs_moyal3 = xr.Dataset(
    {
     'shimg': (shimg_moyal3)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs_mean3 = xr.Dataset(
    {
     'shimg':  (shimg_mean3)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)


#Boundary detection
bi_moyal1 = detect_boundaries_s_M(imgs_moyal1, HiLDA)
bi_moyal2 = detect_boundaries_s_m(imgs_moyal2, HiLDA)
bi_moyal3 = detect_boundaries_s_M(imgs_moyal3, HiLDA)

bi_mean1 = detect_boundaries_s_m(imgs_mean1, HiLDA)
bi_mean2 = detect_boundaries_s_m(imgs_mean2, HiLDA)
bi_mean3 = detect_boundaries_s_m(imgs_mean3, HiLDA)


#Fitting the boundary model
bmF_moyal1 = boundarymodel_F(bi_moyal1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal2 = boundarymodel_F(bi_moyal2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal3 = boundarymodel_F(bi_moyal3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)

bmF_mean1 = boundarymodel_F(bi_mean1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean2 = boundarymodel_F(bi_mean2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean3 = boundarymodel_F(bi_mean3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)

#Removing indexes to only plot from 60 MLAT
moyal_mean1 = nans + shimg_moyal1[1030::]
mean1 = nans + shimg_mean1[1030::]
diff1 = np.subtract(moyal_mean1, mean1) 

moyal_mean2 = nans + shimg_moyal2[1030::]
mean2 = nans + shimg_mean2[1030::]
diff2 = np.subtract(moyal_mean2, mean2) 

moyal_mean3 = nans + shimg_moyal3[1030::]
mean3 = nans + shimg_mean3[1030::]
diff3 = np.subtract(moyal_mean3, mean3) 


#Creates figure and plots polar plots
fig, axs = plt.subplots(3,3, figsize=(12,14))
pax11 = pp(axs[0,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax11.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean1, cmap= 'viridis', norm=norm)
title1 = "Moyal mean "
axs[0,0].set_title(title1)


pax21 = pp(axs[0,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax21.filled_cells(grid[0], grid[1], 1, mlt_arr, mean1, cmap= 'viridis', norm=norm)
title2 = 'Mean'
axs[0,1].set_title(title2)

pax31 = pp(axs[0,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax31.filled_cells(grid[0], grid[1], 1, mlt_arr, diff1, cmap= 'bwr', norm=norm)
title3 = 'Difference'
axs[0,2].set_title(title3)
pax31.writeLTlabels(60, backgroundcolor='white')
pax31.writeLATlabels(color='black')

pax12 = pp(axs[1,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax12.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean2, cmap= 'viridis', norm=norm)

pax22 = pp(axs[1,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax22.filled_cells(grid[0], grid[1], 1, mlt_arr, mean2, cmap= 'viridis', norm=norm)

pax32 = pp(axs[1,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax32.filled_cells(grid[0], grid[1], 1, mlt_arr, diff2, cmap= 'bwr', norm=norm)

pax31 = pp(axs[2,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax31.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean3, cmap= 'viridis', norm=norm)

pax32 = pp(axs[2,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax32.filled_cells(grid[0], grid[1], 1, mlt_arr, mean3, cmap= 'viridis', norm=norm)

pax33 = pp(axs[2,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax33.filled_cells(grid[0], grid[1], 1, mlt_arr, diff3, cmap= 'bwr', norm=norm)

cax2 = plt.axes((0.6, 0.03, 0.4, .01))
cbar2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,  cmap='bwr'), cax= cax2,fraction=0.036, pad=0.04, orientation='horizontal')
cbar2.set_label('Difference [R]', size=12)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
cax1 = plt.axes((0.05, 0.03, 0.5, .01))
cbar1 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,  cmap='viridis'), cax=cax1,fraction=0.036, pad=0.04, orientation='horizontal')
cbar1.set_label('Intensity [R]', size=12)


#Prints file info and file number to the plot
axs[0,0].text(0.0001,0.48,'1' , transform = axs[0,0].transAxes, size=20)
axs[1,0].text(0.0001,0.48,'2' , transform = axs[1,0].transAxes, size=20)
axs[2,0].text(0.0001,0.48,'3' , transform = axs[2,0].transAxes, size=20)

cax3 = plt.axes((1, 0.02, 0.1, 1))
cax3.text(0.0,0.73,f'Bz = {round(float(file_info1["avg_Bz"]),2)}nT\nBy = {round(float(file_info1["avg_By"]),2)}nT\nOrbits = {file_info1["orbits"]}\nVx = {round(float(file_info1["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0,0.43,f'Bz = {round(float(file_info2["avg_Bz"]),2)}nT\nBy = {round(float(file_info2["avg_By"]),2)}nT\nOrbits = {file_info2["orbits"]}\nVx = {round(float(file_info2["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0,0.13,f'Bz = {round(float(file_info3["avg_Bz"]),2)}nT\nBy = {round(float(file_info3["avg_By"]),2)}nT\nOrbits = {file_info3["orbits"]}\nVx = {round(float(file_info3["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.set_axis_off()


#Plots boundaries
fuv.plotboundaries(bmF_moyal1,'pb',pax=pax11,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_moyal1,'eb',pax=pax11,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_moyal2,'pb',pax=pax21,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_moyal2,'eb',pax=pax21,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_moyal3,'pb',pax=pax31,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_moyal3,'eb',pax=pax31,cmap='plasma',add_cbar=False)

fuv.plotboundaries(bmF_mean1,'pb',pax=pax12,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_mean1,'eb',pax=pax12,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_mean2,'pb',pax=pax22,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_mean2,'eb',pax=pax22,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_mean3,'pb',pax=pax32,cmap='plasma',add_cbar=False)
fuv.plotboundaries(bmF_mean3,'eb',pax=pax32,cmap='plasma',add_cbar=False)

#Saves figure
plt.tight_layout(w_pad=-2,h_pad=-6)
plt.savefig(f'plots/lmfit_tilt_{tilt_titl}_{By_titl}_{hemi}_150.pdf', bbox_inches='tight')
