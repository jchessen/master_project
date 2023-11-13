#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:14:59 2023

@author: gaj002
"""

"""This script creates plots of the Moyal mean, total mean, and the difference between them during local winter
    in the Northern Hemisphere for all the datasets. """


import os
# import dipole
import matplotlib
import glob
import numpy as np
import fuvpy
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
# from pysymmetry import visualization
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
filnavn4 = '0100_0130'
filnavn5 = '0131_0160'


#Data selection based on hemisphere, IMF By and dipole tilt angle
hemi = 'north'

#select one IMF B_y at a time
By_val = 'By < -5 nT'
By_titl = '-5'

By_val = 'By > 5 nT'
By_titl = '5'

tilt_titl = -15
tilt_lim = 'tilt < -15'

#Checks for HiLDA
if (By_titl == '5') and (tilt_titl > 0):
    HiLDA = True
else: HiLDA = False



nans = [np.nan for i in range(0,1030)]
date = [datetime(2023, 8, 28, 9, 36), datetime(2023, 8, 28, 9, 46), datetime(2023, 8, 28, 9, 56)]


#Open files
file1 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn1}.hdf5')
file2 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn2}.hdf5')
file3 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn3}.hdf5')
file4 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn4}.hdf5')
file5 = vaex.open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn5}.hdf5')


with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn1}.pickle', 'rb') as handle:
    file_info1 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn2}.pickle', 'rb') as handle:
    file_info2 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn3}.pickle', 'rb') as handle:
    file_info3 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn4}.pickle', 'rb') as handle:
    file_info4 = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_0.04_{filnavn5}.pickle', 'rb') as handle:
    file_info5 = pickle.load(handle)

#Loads intensities from files, adds offset, and creates xarrays
try:
    shimg_moyal1 = list(file1.lbhs_pos.values+100)
    shimg_mean1 = list(file1.lbhs_pos_1.values+100)
    shimg_moyal2 = list(file2.lbhs_pos.values+100)
    shimg_mean2 = list(file2.lbhs_pos_1.values+100)
    shimg_moyal3 = list(file3.lbhs_pos.values+100)
    shimg_mean3 = list(file3.lbhs_pos_1.values+100)
    shimg_moyal4 = list(file4.lbhs_pos.values+100)
    shimg_mean4 = list(file4.lbhs_pos_1.values+100)
    shimg_moyal5 = list(file5.lbhs_pos.values+100)
    shimg_mean5 = list(file5.lbhs_pos_1.values+100)
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
imgs_moyal4 = xr.Dataset(
    {
     'shimg': (shimg_moyal4)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs_mean4 = xr.Dataset(
    {
     'shimg':  (shimg_mean4)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)
imgs_moyal5 = xr.Dataset(
    {
     'shimg': (shimg_moyal5)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs_mean5 = xr.Dataset(
    {
     'shimg':  (shimg_mean5)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

#Boundary detection
bi_moyal1 = detect_boundaries_w_M(imgs_moyal1, HiLDA)
bi_mean1 = detect_boundaries_w_m(imgs_mean1, HiLDA)
bi_moyal2 = detect_boundaries_w_M(imgs_moyal2, HiLDA)
bi_mean2 = detect_boundaries_w_m(imgs_mean2, HiLDA)
bi_moyal3 = detect_boundaries_w_M(imgs_moyal3, HiLDA)
bi_mean3 = detect_boundaries_w_m(imgs_mean3, HiLDA)
bi_moyal4 = detect_boundaries_w_M(imgs_moyal4, HiLDA)
bi_mean4 = detect_boundaries_w_m(imgs_mean4, HiLDA)
bi_moyal5 = detect_boundaries_w_M(imgs_moyal5, HiLDA)
bi_mean5 = detect_boundaries_w_m(imgs_mean5, HiLDA)

#Fitting the boundary model
bmF_moyal1 = boundarymodel_F(bi_moyal1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean1 = boundarymodel_F(bi_mean1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal2 = boundarymodel_F(bi_moyal2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean2 = boundarymodel_F(bi_mean2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal3 = boundarymodel_F(bi_moyal3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean3 = boundarymodel_F(bi_mean3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal4 = boundarymodel_F(bi_moyal4,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean4 = boundarymodel_F(bi_mean4,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_moyal5 = boundarymodel_F(bi_moyal5,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF_mean5 = boundarymodel_F(bi_mean5,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)


#Removing indexes to only plot from 60 MLAT
moyal_mean1 = nans + shimg_moyal1[1030::]
tot_mean1 = nans + shimg_mean1[1030::]
diff_1 = np.subtract(moyal_mean1, tot_mean1) 

moyal_mean2 = nans + shimg_moyal2[1030::]
tot_mean2 = nans + shimg_mean2[1030::]
diff_2 = np.subtract(moyal_mean2, tot_mean2) 

moyal_mean3 = nans + shimg_moyal3[1030::]
tot_mean3 = nans + shimg_mean3[1030::]
diff_3 = np.subtract(moyal_mean3, tot_mean3) 

moyal_mean4 = nans + shimg_moyal4[1030::]
tot_mean4 = nans + shimg_mean4[1030::]
diff_4 = np.subtract(moyal_mean4, tot_mean4) 

moyal_mean5 = nans + shimg_moyal5[1030::]
tot_mean5 = nans + shimg_mean5[1030::]
diff_5 = np.subtract(moyal_mean5, tot_mean5) 


#Creates figure and plots polar plots
fig, axs = plt.subplots(5,3, figsize=(13,22))
moyal1 = pp(axs[0,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
moyal1.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean1, cmap= 'viridis', norm=norm)
title1 = "Moyal mean "
axs[0,0].set_title(title1, size=20)


mean1 = pp(axs[0,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
mean1.filled_cells(grid[0], grid[1], 1, mlt_arr, tot_mean1, cmap= 'viridis', norm=norm)
title2 = 'Mean'
axs[0,1].set_title(title2, size=20)

diff1 = pp(axs[0,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
diff1.filled_cells(grid[0], grid[1], 1, mlt_arr, diff_1, cmap= 'bwr', norm=norm)
title3 = 'Difference'
axs[0,2].set_title(title3, size=20)
diff1.writeLTlabels(60, backgroundcolor='white')
diff1.writeLATlabels(color='black')

moyal2 = pp(axs[1,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
moyal2.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean2, cmap= 'viridis', norm=norm)

mean2 = pp(axs[1,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
mean2.filled_cells(grid[0], grid[1], 1, mlt_arr, tot_mean2, cmap= 'viridis', norm=norm)

diff2 = pp(axs[1,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
diff2.filled_cells(grid[0], grid[1], 1, mlt_arr, diff_2, cmap= 'bwr', norm=norm)

moyal3 = pp(axs[2,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
moyal3.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean3, cmap= 'viridis', norm=norm)

mean3 = pp(axs[2,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
mean3.filled_cells(grid[0], grid[1], 1, mlt_arr, tot_mean3, cmap= 'viridis', norm=norm)

diff3 = pp(axs[2,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
diff3.filled_cells(grid[0], grid[1], 1, mlt_arr, diff_3, cmap= 'bwr', norm=norm)

moyal4 = pp(axs[3,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
moyal4.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean4, cmap= 'viridis', norm=norm)

mean4 = pp(axs[3,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
mean4.filled_cells(grid[0], grid[1], 1, mlt_arr, tot_mean4, cmap= 'viridis', norm=norm)

diff4 = pp(axs[3,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
diff4.filled_cells(grid[0], grid[1], 1, mlt_arr, diff_4, cmap= 'bwr', norm=norm)

moyal5 = pp(axs[4,0], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
moyal5.filled_cells(grid[0], grid[1], 1, mlt_arr, moyal_mean5, cmap= 'viridis', norm=norm)

mean5 = pp(axs[4,1], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
mean5.filled_cells(grid[0], grid[1], 1, mlt_arr, tot_mean5, cmap= 'viridis', norm=norm)

diff5 = pp(axs[4,2], color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = -250
xxx2 = 250
d = (xxx2-xxx1)/250
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.bwr
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
diff5.filled_cells(grid[0], grid[1], 1, mlt_arr, diff_5, cmap= 'bwr', norm=norm)


cax2 = plt.axes((0.665, 0.015, 0.3, .012))
cbar2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,  cmap='bwr'), cax= cax2,fraction=0.036, pad=0.04, orientation='horizontal')
cbar2.set_label('Difference [R]', size=16)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
cax1 = plt.axes((0.1, 0.015, 0.5, .012))
cbar1 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,  cmap='viridis'), cax=cax1,fraction=0.036, pad=0.04, orientation='horizontal')
cbar1.set_label('Intensity [R]', size=16)

#Plots boundaries
plotboundaries(bmF_moyal1,'pb',pax=moyal1,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal1,'eb',pax=moyal1,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal2,'pb',pax=moyal2,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal2,'eb',pax=moyal2,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal3,'pb',pax=moyal3,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal3,'eb',pax=moyal3,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal4,'pb',pax=moyal4,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal4,'eb',pax=moyal4,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal5,'pb',pax=moyal5,cmap='plasma',add_cbar=False)
plotboundaries(bmF_moyal5,'eb',pax=moyal5,cmap='plasma',add_cbar=False)

plotboundaries(bmF_mean1,'pb',pax=mean1,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean1,'eb',pax=mean1,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean2,'pb',pax=mean2,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean2,'eb',pax=mean2,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean3,'pb',pax=mean3,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean3,'eb',pax=mean3,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean4,'pb',pax=mean4,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean4,'eb',pax=mean4,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean5,'pb',pax=mean5,cmap='plasma',add_cbar=False)
plotboundaries(bmF_mean5,'eb',pax=mean5,cmap='plasma',add_cbar=False)


#Prints file info and file number to the plot
axs[0,0].text(0.0001,0.48,'1' , transform = axs[0,0].transAxes, size=20)
axs[1,0].text(0.0001,0.48,'2' , transform = axs[1,0].transAxes, size=20)
axs[2,0].text(0.0001,0.48,'3' , transform = axs[2,0].transAxes, size=20)
axs[3,0].text(0.0001,0.48,'4' , transform = axs[3,0].transAxes, size=20)
axs[4,0].text(0.0001,0.48,'5' , transform = axs[4,0].transAxes, size=20)

cax3 = plt.axes((1, 0.02, 0.1, 1))
cax3.text(0.0,0.825,f'Bz = {round(float(file_info1["avg_Bz"]),2)}nT\nBy = {round(float(file_info1["avg_By"]),2)}nT\nOrbits = {file_info1["orbits"]}\nVx = {round(float(file_info1["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0,0.635,f'Bz = {round(float(file_info2["avg_Bz"]),2)}nT\nBy = {round(float(file_info2["avg_By"]),2)}nT\nOrbits = {file_info2["orbits"]}\nVx = {round(float(file_info2["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0, .45,f'Bz = {round(float(file_info3["avg_Bz"]),2)}nT\nBy = {round(float(file_info3["avg_By"]),2)}nT\nOrbits = {file_info3["orbits"]}\nVx = {round(float(file_info3["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0, .26,f'Bz = {round(float(file_info4["avg_Bz"]),2)}nT\nBy = {round(float(file_info4["avg_By"]),2)}nT\nOrbits = {file_info4["orbits"]}\nVx = {round(float(file_info4["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.text(0.0, 0.07,f'Bz = {round(float(file_info5["avg_Bz"]),2)}nT\nBy = {round(float(file_info5["avg_By"]),2)}nT\nOrbits = {file_info5["orbits"]}\nVx = {round(float(file_info5["avg_Vx"]))} km/s' , transform = cax3.transAxes, size=20)
cax3.set_axis_off()

#Saves figure
plt.tight_layout(w_pad=-2,h_pad=-6)
plt.savefig(f'plots/lmfit_{By_titl}_tilt_{tilt_titl}.pdf',bbox_inches='tight')
