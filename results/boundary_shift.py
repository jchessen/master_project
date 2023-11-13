#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:05:42 2023

@author: gaj002
"""

"""This script plots the estimated auroral boundaries and calculates the centers of the polar caps."""

import os
# import dipole
import matplotlib
import glob
import numpy as np
import fuvpy as fuv
from fuvpy.src.boundaries import boundarymodel_F
from fuvpy.src.plotting import plotboundaries
from fuvpy.utils import spherical as sp
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
import great_circle_calculator.great_circle_calculator as gcc
from functions import *
from scipy import stats


#Define grid
dr = 1
M0=5
lowlat=50
grid, mlt_arr = sdarngrid(dlat=1, dlon=2, latmin=50)

#select file
filnavn = '0061_0099'
number = 3
hemi = 'north'


By_val = 'By < -5 nT'
By_titl2 = '-5'

By_val = 'By > 5 nT'
By_titl1 = '5'


tilt_titl2 = -15
tilt_lim2 = 'tilt < -15'

tilt_titl1 = 15
tilt_lim1 = 'tilt > 15'

nans = [np.nan for i in range(0,1030)]

#opens files
with open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl1}_cvar_0.04_{filnavn}.pickle', 'rb') as handle:
    file_1_info = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl2}_cvar_0.04_{filnavn}.pickle', 'rb') as handle:
    file_2_info = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl1}_cvar_0.04_{filnavn}.pickle', 'rb') as handle:
    file_3_info = pickle.load(handle)
with open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl2}_cvar_0.04_{filnavn}.pickle', 'rb') as handle:
    file_4_info = pickle.load(handle)

file_1 = vaex.open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl1}_cvar_0.04_{filnavn}.hdf5')
file_2 = vaex.open(f'data/lmfit_tilt_{tilt_titl1}_{By_titl2}_cvar_0.04_{filnavn}.hdf5')

file_3 = vaex.open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl1}_cvar_0.04_{filnavn}.hdf5')
file_4 = vaex.open(f'data/lmfit_tilt_{tilt_titl2}_{By_titl2}_cvar_0.04_{filnavn}.hdf5')

date = [datetime(2023, 8, 28, 9, 36), datetime(2023, 8, 28, 9, 46), datetime(2023, 8, 28, 9, 56)]

try:
    shimg1 = list(file_1.lbhs_pos.values+150)
    shimg2 = list(file_2.lbhs_pos.values+150)
    shimg3 = list(file_3.lbhs_pos_1.values+100)
    shimg4 = list(file_4.lbhs_pos_1.values+100)
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

imgs3 = xr.Dataset(
    {
     'shimg': (shimg3)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

imgs4 = xr.Dataset(
    {
     'shimg':  (shimg4)
    },
    coords={
        'mlat': (grid[0]),
        'mlt':(grid[1]),
        'date': date
        
        },
)

#estimates boundaries
bi1 = detect_boundaries_s_M(imgs1, HiLDA=True)
bi2 = detect_boundaries_s_M(imgs2, HiLDA=False)
bi3 = detect_boundaries_w_m(imgs3, HiLDA=False)
bi4 = detect_boundaries_w_m(imgs4, HiLDA=False)

bmF1 = boundarymodel_F(bi1,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF2 = boundarymodel_F(bi2,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF3 = boundarymodel_F(bi3,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
bmF4 = boundarymodel_F(bi4,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)

#creates mlt-array
lt = bmF1.mlt.values 

#Creates undertainty arrays
pbs1,ebs1,fluxs1 = bound_uncert(bi1)
pbs2,ebs2,fluxs2 = bound_uncert(bi2)
pbs3,ebs3,fluxs3 = bound_uncert(bi3)
pbs4,ebs4,fluxs4 = bound_uncert(bi4)

#Calculates standard deviation
std_p1 = uncert_bound_std(pbs1)
std_e1 = uncert_bound_std(ebs1)
std_p2 = uncert_bound_std(pbs2)
std_e2 = uncert_bound_std(ebs2)

std_p3 = uncert_bound_std(pbs3)
std_e3 = uncert_bound_std(ebs3)
std_p4 = uncert_bound_std(pbs4)
std_e4 = uncert_bound_std(ebs4)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
pax1 = pp(ax1, color='black', linestyle='-', minlat = 60,  plotgrid=True, sector='all', linewidth = 0.5, alpha=0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
pax2 = pp(ax2, color='black', linestyle='-', minlat = 60, linewidth = 0.5)
xxx1 = 0
xxx2 = 1000
d = (xxx2-xxx1)/200
levels = np.array([j for j in np.arange(xxx1,xxx2,d)])
cmap = plt.cm.viridis
norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)

title1 = f'Moyal boundaries ${tilt_lim1}$'
title2 = f'Mean boundaries ${tilt_lim2}$'

#Plots standard deviation of boundaries
uncert_bounds_plotter_pb(pax1,bmF1, std_p1, lt, 'red')
# uncert_bounds_plotter_eb(pax1,bmF1, std_e1, lt, 'red')
uncert_bounds_plotter_pb(pax1,bmF2, std_p2, lt, 'blue')
# uncert_bounds_plotter_eb(pax1,bmF2, std_e2, lt, 'blue')

uncert_bounds_plotter_pb(pax2, bmF3, std_p3, lt, 'red')
# uncert_bounds_plotter_eb(pax2, bmF3, std_e3, lt, 'red')
uncert_bounds_plotter_pb(pax2, bmF4, std_p4, lt, 'blue')
# uncert_bounds_plotter_eb(pax2, bmF4, std_e4, lt, 'blue')


#Can plot all attempts
# uncert_plotter_pos(pax1,pbs1, ebs1,lt)
# uncert_plotter_neg(pax1, pbs2, ebs2,lt)
# uncert_plotter_pos(pax2, pbs3, ebs3,lt)
# uncert_plotter_neg(pax2, pbs4, ebs4,lt)

#Calculates distances and plots center of polar cap
dists1 = distfunc(bmF1['pb'][0].values, bmF1['mlt'].values)
dists2 = distfunc(bmF2['pb'][0].values, bmF2['mlt'].values)
center1 = pp._latlt2xy(pax1,lat=bmF1['pb'][0].values, lt=bmF1['mlt'].values)
center1 = [np.average(center1[0], weights=dists1), np.average(center1[1], weights=dists1)]
center1 = pp._xy2latlt(pax1, center1[0], center1[1])

center2 = pp._latlt2xy(pax1,lat=bmF2['pb'][0].values, lt=bmF2['mlt'].values)
center2 = [np.average(center2[0], weights=dists2), np.average(center2[1], weights=dists2)]
center2 = pp._xy2latlt(pax1, center2[0], center2[1])

pax1.scatter(center1[0], center1[1], color='red', marker='x')
pax1.scatter(center2[0], center2[1], color='blue', marker='x')


dists3 = distfunc(bmF3['pb'][0].values, bmF3['mlt'].values)
dists4 = distfunc(bmF4['pb'][0].values, bmF4['mlt'].values)
center3 = pp._latlt2xy(pax2,lat=bmF3['pb'][0].values, lt=bmF3['mlt'].values)
center3 = [np.average(center3[0], weights=dists3), np.average(center3[1], weights=dists3)]
center3 = pp._xy2latlt(pax2, center3[0], center3[1])

center4 = pp._latlt2xy(pax2,lat=bmF4['pb'][0].values, lt=bmF4['mlt'].values)
center4 = [np.average(center4[0], weights=dists4), np.average(center4[1], weights=dists4)]
center4 = pp._xy2latlt(pax2, center4[0], center4[1])


pax2.scatter(center3[0], center3[1], color='red', marker='x')
pax2.scatter(center4[0],center4[1], color='blue', marker='x')

#Plots boundaries and saves figure
plotboundaries(bmF1,'pb',pax=pax1,cmap='prism_r',add_cbar=False)
plotboundaries(bmF1,'eb',pax=pax1,cmap='prism_r',add_cbar=False)


plotboundaries(bmF2,'pb',pax=pax1,cmap='plasma_r',add_cbar=False, label='By $<$ -5nT')
plotboundaries(bmF2,'eb',pax=pax1,cmap='plasma_r',add_cbar=False)

plotboundaries(bmF3,'pb',pax=pax2,cmap='prism_r',add_cbar=False,label='By $>$ 5nT')
plotboundaries(bmF3,'eb',pax=pax2,cmap='prism_r',add_cbar=False)

plotboundaries(bmF4,'pb',pax=pax2,cmap='plasma_r',add_cbar=False,label='By $<$ -5nT')
plotboundaries(bmF4,'eb',pax=pax2,cmap='plasma_r',add_cbar=False)

ax1.text(0.0001,0.48,number , transform = ax1.transAxes, size=20)
plt.tight_layout(w_pad=-6, h_pad=0)
pax2.writeLATlabels(color='black')
pax2.writeLTlabels(60)
ax2.legend()
cax3 = plt.axes((0.1, 0.99, 0.75, 0.1))
cax3.text(0.175,0.01,'Tilt $>$ 15' , transform = cax3.transAxes, size=20)
cax3.text(0.775,0.001,'Tilt $<$ -15' , transform = cax3.transAxes, size=20)
cax3.set_axis_off()
plt.savefig(f'plots/lmfit_tilt_{filnavn}_{hemi}_boundaries.pdf',bbox_inches='tight')

