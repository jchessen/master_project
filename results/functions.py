#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:50:42 2023

@author: gaj002
"""

"""This file contains the functions used in the handling and plotting of the results"""

import os
# import dipole
import matplotlib
import glob
import numpy as np
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
import great_circle_calculator.great_circle_calculator as gcc
from polplot import grids
import pickle
from pyswipe import SWIPE
from pyswipe.plot_utils import equal_area_grid, Polarplot, get_h2d_bin_areas
from fuvpy.src.boundaries import boundarymodel_F
from fuvpy.src.plotting import plotboundaries
from fuvpy.utils import spherical as sp
from math import atan, pi

#Calulates dayside reconnection rate. equation 14 from Milan et al (2012)
def rec_func(By,Bz,Vx):
    result = []
    for i in range(0,5):
        # By_vec = [float(By[i]),0]
        # Bz_vec = [0, float(Bz[i])]
        theta = atan(float(By[i]/Bz[i]))
        result.append(((3.3*float(1e5)*float(-Vx[i]*1_000)**(4/3))*np.sqrt(float(By[i]*float(1e-9))**2+float(Bz[i]*float(1e-9))**2)*(np.sin((1/2)*abs(theta)))**(9/2))/1_000)
    return result


#calculates standard deviation of all boundary estimation attempts
def uncert_bound_std(bs):
    test1 = []
    std1 = []
    for j in range(0,len(lt)):
        for i in range(len(bs)):
            test1.append(bs[i][j])
            
        std1.append((np.std(test1)))
    return std1
    

#Creates array of all boundary estimation attempts
def bound_uncert(b):
    pbs=[]
    ebs=[]
    fluxs = []
    for i in range(b.pb.shape[2]):
        imgs = b.isel(lim=i)
        imgs = imgs.expand_dims('lim')
        items = boundarymodel_F(imgs,tLeb=1e0,tLpb=1e0,Leb=1e0,Lpb=1e0)
        pbs.append(items.pb.values[1,:])
        ebs.append(items.eb.values[1,:])
        fluxs.append(sum(items.dA.values[1,:])*(np.pi/180)*1.5)
    return pbs,ebs,fluxs

#Plots the uncertainties either using the standard deviation or plotting each attempt (max/min)
def uncert_plotter_pos(pax, pb,eb,lt):
    for i in range(0,len(pb)):    
        pp.plot(pax, pb[i], lt, color='red', alpha=0.05, linewidth=2)
    for i in range(0,len(eb)):
        pp.plot(pax, eb[i], lt, color = 'red', alpha = 0.05, linewidth=2)
        
def uncert_plotter_neg(pax,pb,eb,lt):
    for i in range(0,len(pb)):    
        pp.plot(pax, pb[i], lt, color='blue', alpha=0.05, linewidth=2)
    for i in range(0,len(eb)):
        pp.plot(pax, eb[i], lt, color = 'blue', alpha = 0.05, linewidth=2)

def uncert_bounds_plotter_pb(pax,bmF, std,lt, col):
    for i in range(0,len(lt)-1):
        if i == len(lt):
            cell1 = np.array([np.add(bmF.pb.values[1][i-1], std[i-1]),np.subtract(bmF.pb.values[1][i-1], std[i-1])])
            cell2 = np.array([np.add(bmF.pb.values[1][0], std[0]),np.subtract(bmF.pb.values[1][0], std[0])])
        cell1 = np.array([np.add(bmF.pb.values[1][i], std[i]),np.subtract(bmF.pb.values[1][i], std[i])])
        cell2 = np.array([np.add(bmF.pb.values[1][i+1], std[i+1]),np.subtract(bmF.pb.values[1][i+1], std[i+1])])
        pp.fill(pax, np.array([cell1,cell2]) , np.array([[lt[i],lt[i]],[lt[i+1],lt[i+1]]]), color=col, alpha=0.05, linewidth=5)
def uncert_bounds_plotter_eb(pax, bmF, std,lt, col):
    for i in range(0,len(lt)-1):
        cell1 = np.array([np.add(bmF.eb.values[1][i], std[i]),np.subtract(bmF.eb.values[1][i], std[i])])
        cell2 = np.array([np.add(bmF.eb.values[1][i+1], std[i+1]),np.subtract(bmF.eb.values[1][i+1], std[i+1])])
        pp.fill(pax, np.array([cell1,cell2]) , np.array([[lt[i],lt[i]],[lt[i+1],lt[i+1]]]), color=col, alpha=0.05)

#Calculates distance between each MLT-step (must change altitude in gcc to be R_E + 130km)
def distfunc(lat, mlt):
    dists = []
    for i in range(-1,240):
        if i == 240:
            dist = gcc.distance_between_points(((mlt[0]-12)*15,lat[0]), ((mlt[i]-12)*15,lat[i]),unit='meters', haversine=True)
        else:
            dist = gcc.distance_between_points(((mlt[i]-12)*15,lat[i]), ((mlt[i+1]-12)*15,lat[i+1]),unit='meters', haversine=True)
        dists.append(dist)
    return dists
    
#These functions detects boundaries with different thresholds depending on if the Moyal mean or total was used as well as the local season
#This is denoted by s for summer, w for winter and small and large m for the total mean and Moyal mean respectively.
def detect_boundaries_s_M(imgs, HiLDA,**kwargs):
    '''
    A function to identify auroral boundaries along latitudinal meridians

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimg'.
    lims : array_like, optional
        Count limits to detect boundaries. Default is np.arange(50,201,5)
    sigma : float, optional
        Distance in km from points on the meridian a pixel can be to be included in the median intensity profile. Default is 300 km.
    height : float, optional
        Assumed height of the emissions in km. Default is 130 
    clat_ev : array_like, optional
        Evalutation points of the latitudinal intensity profiles. Default is np.arange(0.5,50.5,0.5)
    mlt_ev : array_like, optional
        Which mlts to evaluate the latitudinal profiles. Default is np.arange(0.5,24,1)

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(150+150,251+150,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,1)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)

    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])
    
    dfs=[]
    for t in range(len(imgs.date)):
        img = imgs.isel(date=t)
    
        # Data coordinates in cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img[inImg].values
    
        # Make latitudinal intensity profiles
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])

        # Make dataset with meridian intensity profiles
        ds = xr.Dataset(coords={'clat':clat_ev,'mlt':mlt_ev,'lim':lims})
        ds['d'] = (['clat','mlt'],d_ev)
        
        # Set values outside outer ring to nan
        ds['d'] = xr.where(ds['clat']>40+10*np.cos(np.pi*ds['mlt']/12),np.nan,ds['d'])
        
        ds['above'] = (ds['d']>ds['lim']).astype(float) 
        ds['above'] = xr.where(np.isnan(ds['d']),np.nan,ds['above'])
        
        diff = ds['above'].diff(dim='clat')
        
        # Find first above
        mask = diff==1
        ds['firstAbove'] = xr.where(mask.any(dim='clat'), mask.argmax(dim='clat'), np.nan)+1
        
        # Find last above
        mask = diff==-1
        val = len(diff.clat) - mask.isel(clat=slice(None,None,-1)).argmax(dim='clat') - 1
        ds['lastAbove']= xr.where(mask.any(dim='clat'), val, np.nan)
    
        # Identify poleward boundaries
        ind = ds['firstAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['firstAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatBelow'] = ds.isel(clat=ind.values-1, drop=True)['clat'].values
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        df['pb'] = np.average(df[['clatBelow','clatAbove']],weights=abs(df[['dAbove','dBelow']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_pb = df[['date','mlt','lim','pb']].set_index(['date','mlt','lim'])    
        
        #Exclude HiLDA-spot
        if HiLDA is True:
            use = ~((df_pb.index.get_level_values('mlt') > 11) & (df_pb.index.get_level_values('mlt') < 17))
            df_pb = df_pb[use]
            
        
        # Identify equatorward boundaries
        ind = ds['lastAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['lastAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        df['clatBelow'] = ds.isel(clat=ind.values+1)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        df['eb'] = np.average(df[['clatAbove','clatBelow']],weights=abs(df[['dBelow','dAbove']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_eb = df[['date','mlt','lim','eb']].set_index(['date','mlt','lim'])  
    
        dfs.append(pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer'))
    
    df = pd.concat(dfs)
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds
def detect_boundaries_s_m(imgs, HiLDA,**kwargs):
    '''
    A function to identify auroral boundaries along latitudinal meridians

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimg'.
    lims : array_like, optional
        Count limits to detect boundaries. Default is np.arange(50,201,5)
    sigma : float, optional
        Distance in km from points on the meridian a pixel can be to be included in the median intensity profile. Default is 300 km.
    height : float, optional
        Assumed height of the emissions in km. Default is 130 
    clat_ev : array_like, optional
        Evalutation points of the latitudinal intensity profiles. Default is np.arange(0.5,50.5,0.5)
    mlt_ev : array_like, optional
        Which mlts to evaluate the latitudinal profiles. Default is np.arange(0.5,24,1)

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(70+150,251+150,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,1)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)

    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])
    
    dfs=[]
    for t in range(len(imgs.date)):
        img = imgs.isel(date=t)
    
        # Data coordinates in cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img[inImg].values
    
        # Make latitudinal intensity profiles
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])

        # Make dataset with meridian intensity profiles
        ds = xr.Dataset(coords={'clat':clat_ev,'mlt':mlt_ev,'lim':lims})
        ds['d'] = (['clat','mlt'],d_ev)
        
        # Set values outside outer ring to nan
        ds['d'] = xr.where(ds['clat']>40+10*np.cos(np.pi*ds['mlt']/12),np.nan,ds['d'])
        
        ds['above'] = (ds['d']>ds['lim']).astype(float) 
        ds['above'] = xr.where(np.isnan(ds['d']),np.nan,ds['above'])
        
        diff = ds['above'].diff(dim='clat')
        
        # Find first above
        mask = diff==1
        ds['firstAbove'] = xr.where(mask.any(dim='clat'), mask.argmax(dim='clat'), np.nan)+1
        
        # Find last above
        mask = diff==-1
        val = len(diff.clat) - mask.isel(clat=slice(None,None,-1)).argmax(dim='clat') - 1
        ds['lastAbove']= xr.where(mask.any(dim='clat'), val, np.nan)
    
        # Identify poleward boundaries
        ind = ds['firstAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['firstAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatBelow'] = ds.isel(clat=ind.values-1, drop=True)['clat'].values
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        df['pb'] = np.average(df[['clatBelow','clatAbove']],weights=abs(df[['dAbove','dBelow']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_pb = df[['date','mlt','lim','pb']].set_index(['date','mlt','lim'])    
        
        #Exclude HiLDA-spot
        if HiLDA is True:
            use = ~((df_pb.index.get_level_values('mlt') > 11) & (df_pb.index.get_level_values('mlt') < 17))
            df_pb = df_pb[use]
            
        
        # Identify equatorward boundaries
        ind = ds['lastAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['lastAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        df['clatBelow'] = ds.isel(clat=ind.values+1)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        df['eb'] = np.average(df[['clatAbove','clatBelow']],weights=abs(df[['dBelow','dAbove']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_eb = df[['date','mlt','lim','eb']].set_index(['date','mlt','lim'])  
    
        dfs.append(pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer'))
    
    df = pd.concat(dfs)
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds

def detect_boundaries_w_M(imgs, HiLDA,**kwargs):
    '''
    A function to identify auroral boundaries along latitudinal meridians

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimg'.
    lims : array_like, optional
        Count limits to detect boundaries. Default is np.arange(50,201,5)
    sigma : float, optional
        Distance in km from points on the meridian a pixel can be to be included in the median intensity profile. Default is 300 km.
    height : float, optional
        Assumed height of the emissions in km. Default is 130 
    clat_ev : array_like, optional
        Evalutation points of the latitudinal intensity profiles. Default is np.arange(0.5,50.5,0.5)
    mlt_ev : array_like, optional
        Which mlts to evaluate the latitudinal profiles. Default is np.arange(0.5,24,1)

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(70+100,250+100,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,1)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)

    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])
    
    dfs=[]
    for t in range(len(imgs.date)):
        img = imgs.isel(date=t)
    
        # Data coordinates in cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img[inImg].values
    
        # Make latitudinal intensity profiles
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])

        # Make dataset with meridian intensity profiles
        ds = xr.Dataset(coords={'clat':clat_ev,'mlt':mlt_ev,'lim':lims})
        ds['d'] = (['clat','mlt'],d_ev)
        
        # Set values outside outer ring to nan
        ds['d'] = xr.where(ds['clat']>40+10*np.cos(np.pi*ds['mlt']/12),np.nan,ds['d'])
        
        ds['above'] = (ds['d']>ds['lim']).astype(float) 
        ds['above'] = xr.where(np.isnan(ds['d']),np.nan,ds['above'])
        
        diff = ds['above'].diff(dim='clat')
        
        # Find first above
        mask = diff==1
        ds['firstAbove'] = xr.where(mask.any(dim='clat'), mask.argmax(dim='clat'), np.nan)+1
        
        # Find last above
        mask = diff==-1
        val = len(diff.clat) - mask.isel(clat=slice(None,None,-1)).argmax(dim='clat') - 1
        ds['lastAbove']= xr.where(mask.any(dim='clat'), val, np.nan)
    
        # Identify poleward boundaries
        ind = ds['firstAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['firstAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatBelow'] = ds.isel(clat=ind.values-1, drop=True)['clat'].values
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        df['pb'] = np.average(df[['clatBelow','clatAbove']],weights=abs(df[['dAbove','dBelow']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_pb = df[['date','mlt','lim','pb']].set_index(['date','mlt','lim'])    
        
        #Exclude HiLDA-spot
        if HiLDA is True:
            use = ~((df_pb.index.get_level_values('mlt') > 11) & (df_pb.index.get_level_values('mlt') < 17))
            df_pb = df_pb[use]
            
        
        # Identify equatorward boundaries
        ind = ds['lastAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['lastAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        df['clatBelow'] = ds.isel(clat=ind.values+1)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        df['eb'] = np.average(df[['clatAbove','clatBelow']],weights=abs(df[['dBelow','dAbove']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_eb = df[['date','mlt','lim','eb']].set_index(['date','mlt','lim'])  
    
        dfs.append(pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer'))
    
    df = pd.concat(dfs)
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds

def detect_boundaries_w_m(imgs, HiLDA,**kwargs):
    '''
    A function to identify auroral boundaries along latitudinal meridians

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimg'.
    lims : array_like, optional
        Count limits to detect boundaries. Default is np.arange(50,201,5)
    sigma : float, optional
        Distance in km from points on the meridian a pixel can be to be included in the median intensity profile. Default is 300 km.
    height : float, optional
        Assumed height of the emissions in km. Default is 130 
    clat_ev : array_like, optional
        Evalutation points of the latitudinal intensity profiles. Default is np.arange(0.5,50.5,0.5)
    mlt_ev : array_like, optional
        Which mlts to evaluate the latitudinal profiles. Default is np.arange(0.5,24,1)

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(70+100,150+100,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,1)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)

    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])
    
    dfs=[]
    for t in range(len(imgs.date)):
        img = imgs.isel(date=t)
    
        # Data coordinates in cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img[inImg].values
    
        # Make latitudinal intensity profiles
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])

        # Make dataset with meridian intensity profiles
        ds = xr.Dataset(coords={'clat':clat_ev,'mlt':mlt_ev,'lim':lims})
        ds['d'] = (['clat','mlt'],d_ev)
        
        # Set values outside outer ring to nan
        ds['d'] = xr.where(ds['clat']>40+10*np.cos(np.pi*ds['mlt']/12),np.nan,ds['d'])
        
        ds['above'] = (ds['d']>ds['lim']).astype(float) 
        ds['above'] = xr.where(np.isnan(ds['d']),np.nan,ds['above'])
        
        diff = ds['above'].diff(dim='clat')
        
        # Find first above
        mask = diff==1
        ds['firstAbove'] = xr.where(mask.any(dim='clat'), mask.argmax(dim='clat'), np.nan)+1
        
        # Find last above
        mask = diff==-1
        val = len(diff.clat) - mask.isel(clat=slice(None,None,-1)).argmax(dim='clat') - 1
        ds['lastAbove']= xr.where(mask.any(dim='clat'), val, np.nan)
    
        # Identify poleward boundaries
        ind = ds['firstAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['firstAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatBelow'] = ds.isel(clat=ind.values-1, drop=True)['clat'].values
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        df['pb'] = np.average(df[['clatBelow','clatAbove']],weights=abs(df[['dAbove','dBelow']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_pb = df[['date','mlt','lim','pb']].set_index(['date','mlt','lim'])    
        
        #Exclude HiLDA-spot
        if HiLDA is True:
            use = ~((df_pb.index.get_level_values('mlt') > 11) & (df_pb.index.get_level_values('mlt') < 17))
            df_pb = df_pb[use]
            
        
        # Identify equatorward boundaries
        ind = ds['lastAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['lastAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index(drop=True)
        # df= ind.to_dataframe().drop(columns=['lim','mlt']).reset_index()
        df['clatAbove'] = ds.isel(clat=ind.values)['clat'].values
        df['clatBelow'] = ds.isel(clat=ind.values+1)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        df['eb'] = np.average(df[['clatAbove','clatBelow']],weights=abs(df[['dBelow','dAbove']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_eb = df[['date','mlt','lim','eb']].set_index(['date','mlt','lim'])  
    
        dfs.append(pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer'))
    
    df = pd.concat(dfs)
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds


#Our edited function to create potential plots. Original function can be found in the SWIPE package
def plot_potential_(sw1,sw2,tit1,tit2,title,filnavn,bmF1,bmF2,tilt_lim,hemi,tilt_titl,number,
                   convection=False,
                   vector_scale=None,
                   minlat_for_cpcp_calc = 65,
                   flip_panel_order=False,
                   vmin=None,
                   vmax=None):
    """ 
    Create a summary plot of the ionospheric potential and electric field

    Parameters
    ----------
    convection           : boolean (default False)
        Show convection velocity (in m/s) instead of convection electric field
    vector_scale         : optional
        Vector lengths are shown relative to a template. This parameter determines
        the magnitude of that template, in mV/m. Default is 20 mV/m
    minlat_for_cpcp_calc : float (default 65)
        Minimum latitude allowed for determination of min/max potential when 
        calculating the cross-polar cap potential

    Examples
    --------
    >>> # initialize by supplying a set of external conditions:
    >>> m = SWIPE(300, # solar wind velocity in km/s 
                 -4, # IMF By in nT
                 -3, # IMF Bz in nT
                 20, # dipole tilt angle in degrees
                 150) # F10.7 index in s.f.u.
    >>> # make summary plot:
    >>> m.plot_potential()

    """

    # get the grids:
    mlats, mlts = sw1.plotgrid_scalar
    mlatv, mltv = sw1.plotgrid_vector

    # set up figure and polar coordinate plots:
    fig = plt.figure(figsize = (15, 7))
    if flip_panel_order:
        pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **sw2.pax_plotopts)
        pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **sw1.pax_plotopts)
    else:
        pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **sw1.pax_plotopts)
        pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **sw2.pax_plotopts)
    pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
    
    # labels
    pax_n.writeLTlabels(lat = sw1.minlat, size = 14)
    pax_s.writeLTlabels(lat = sw2.minlat, size = 14)
    pax_n.write(sw1.minlat, 3,    str(sw1.minlat) + r'$^\circ$' , ha = 'left', va = 'top', size = 14)
    pax_s.write(sw2.minlat, 3,    str(sw2.minlat) + '$^\circ$', ha = 'left', va = 'top', size = 14)
    pax_n.write(sw1.minlat-5, 18, f'{number}' , ha = 'center', va = 'center', size = 22, ignore_plot_limits=True)
    # pax_s.write(self.minlat-5, 12, r'South' , ha = 'center', va = 'center', size = 18)
    # sw1._make_pax_n_label(pax_n)
    # sw2._make_pax_s_label(pax_s)

    # OLD
    # # calculate and plot FAC
    # Jun, Jus = np.split(self.get_upward_current(), 2)
    # faclevels = np.r_[-.925:.926:.05]
    # pax_n.contourf(mlats, mlts, Jun, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')
    # pax_s.contourf(mlats, mlts, Jus, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')


    # NEW
    # calculate and plot potential
    phinsw1, phis = np.split(sw1.get_potential(), 2)
    phinsw2, phis = np.split(sw2.get_potential(), 2)
    phin = phinsw1 - np.median(phinsw1)
    phis = phinsw2 - np.median(phinsw2)
    # potlevels = np.r_[-.4:.4:.025]
    # potlevels = np.r_[-8.5:8.6:1]

    # minpot,maxpot = -25.5,26
    # potlevelscoarse = np.r_[minpot:maxpot:5]
    # potlevels = np.r_[minpot:maxpot:.25]

    if vmin is None and vmax is None:
        minpot = np.min([np.quantile(phin,0.03),np.quantile(phis,0.03)])
        maxpot = np.max([np.quantile(phin,0.97),np.quantile(phis,0.97)])

        maxpot = np.max(np.abs([minpot,maxpot]))
        minpot = -maxpot

    else:
        if vmin is not None:
            minpot = vmin
        else:
            minpot = np.min([np.quantile(phin,0.03),np.quantile(phis,0.03)])
        
        if vmax is not None:            
            maxpot = vmax
        else:
            maxpot = np.max([np.quantile(phin,0.97),np.quantile(phis,0.97)])

    potlevelscoarse = np.r_[minpot:maxpot:5]
    potlevels = np.r_[minpot:maxpot:.25]

    pax_n.contourf(mlats, mlts, phin, levels = potlevels, cmap = plt.cm.bwr, extend = 'both')
    pax_s.contourf(mlats, mlts, phis, levels = potlevels, cmap = plt.cm.bwr, extend = 'both')

    opts__contour = dict(levels = potlevelscoarse, linestyles='solid', colors='black', linewidths=1)
    pax_n.contour(mlats, mlts, phin, **opts__contour)
    pax_s.contour(mlats, mlts, phis, **opts__contour)

    # OLD (no limit on minlat for cpcp)
    dPhiN = phin.max()-phin.min()
    dPhiS = phis.max()-phis.min()
    minn,maxn = np.argmin(phin),np.argmax(phin)
    mins,maxs = np.argmin(phis),np.argmax(phis)
    cpcpmlats,cpcpmlts = mlats,mlts

    # NEW (yes limit on minlat for cpcp)        
    OKinds = mlats.flatten() > minlat_for_cpcp_calc
    cpcpmlats,cpcpmlts = mlats.flatten()[OKinds],mlts.flatten()[OKinds]
    dPhiN = phin[OKinds].max()-phin[OKinds].min()
    dPhiS = phis[OKinds].max()-phis[OKinds].min()
    minn,maxn = np.argmin(phin[OKinds]),np.argmax(phin[OKinds])
    mins,maxs = np.argmin(phis[OKinds]),np.argmax(phis[OKinds])

    pax_n.write(sw1.minlat-6, 2, r'$\Delta \Phi = $'+f"{dPhiN:.1f} kV" ,
                ha='center',va='center',size=18,ignore_plot_limits=True)
    pax_s.write(sw2.minlat-6, 2, r'$\Delta \Phi = $'+f"{dPhiS:.1f} kV" ,
                ha='center',va='center',size=18,ignore_plot_limits=True)
    
    pax_n.write(cpcpmlats[minn],cpcpmlts[minn],r'x',
                ha='center',va='center',size=18,ignore_plot_limits=True)
    pax_n.write(cpcpmlats[maxn],cpcpmlts[maxn],r'+',
                ha='center',va='center',size=18,ignore_plot_limits=True)

    pax_s.write(cpcpmlats[mins],cpcpmlts[mins],r'x',
                ha='center',va='center',size=18,ignore_plot_limits=True)
    pax_s.write(cpcpmlats[maxs],cpcpmlts[maxs],r'+',
                ha='center',va='center',size=18,ignore_plot_limits=True)

    # colorbar
    pax_c.contourf(np.vstack((np.zeros_like(potlevels), np.ones_like(potlevels))), 
                   np.vstack((potlevels, potlevels)), 
                   np.vstack((potlevels, potlevels)), 
                   levels = potlevels, cmap = plt.cm.bwr)
    pax_c.set_xticks([])
    # pax_c.set_ylabel(r'downward    $\mu$A/m$^2$      upward', size = 18)
    pax_c.set_ylabel(r'neg    kV      pos', size = 18)
    pax_c.yaxis.set_label_position("right")
    pax_c.yaxis.tick_right()

    # print AL index values and integrated up/down currents
    # AL_n, AL_s, AU_n, AU_s = self.get_AE_indices()
    # ju_n, jd_n, ju_s, jd_s = self.get_integrated_upward_current()

    # pax_n.ax.text(pax_n.ax.get_xlim()[0], pax_n.ax.get_ylim()[0], 
    #               'AL: \t${AL_n:+}$ nT\nAU: \t${AU_n:+}$ nT\n $\int j_{uparrow:}$:\t ${jn_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${jn_down:+.1f}$ MA'.format(AL_n = int(np.round(AL_n)), AU_n = int(np.round(AU_n)), jn_up = ju_n, jn_down = jd_n, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)
    # pax_s.ax.text(pax_s.ax.get_xlim()[0], pax_s.ax.get_ylim()[0], 
    #               'AL: \t${AL_s:+}$ nT\nAU: \t${AU_s:+}$ nT\n $\int j_{uparrow:}$:\t ${js_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${js_down:+.1f}$ MA'.format(AL_s = int(np.round(AL_s)), AU_s = int(np.round(AU_s)), js_up = ju_s, js_down = jd_s, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)

    plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

    pax_n.ax.set_title(f'${tit1}$', size=16)
    pax_s.ax.set_title(f'${tit2}$', size=16)
    plt.suptitle(title + f'\n${tilt_lim}$', size=20)
    
    _ = sw1._make_figtitle(fig,x=0.02,y=0.07,ha='center',size=16)
    _ = sw2._make_figtitle(fig,x=0.48,y=0.07,ha='center',size=16)
    plotboundaries(bmF1,'pb',pax=pax_n,cmap='prism',add_cbar=False, linewidth=3)
    plotboundaries(bmF1,'eb',pax=pax_n,cmap='prism',add_cbar=False, linewidth=3)
    
    plotboundaries(bmF2,'pb',pax=pax_s,cmap='prism',add_cbar=False, linewidth=3)
    plotboundaries(bmF2,'eb',pax=pax_s,cmap='prism',add_cbar=False, linewidth=3)
    plt.savefig(f'{hemi}/pdf/master_plots/results/{filnavn}/lmfit_tilt_{tilt_titl}_conv_test.pdf')
    plt.savefig(f'{hemi}/pdf/master_plots/results/{filnavn}/lmfit_tilt_{tilt_titl}_conv_test.svg')

    # plt.show()

    return fig, (pax_n, pax_s, pax_c)


