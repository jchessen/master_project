#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:29:41 2023

@author: gaj002
"""
"""This script filters the data according to dipole tilt angle and IMF By and Bz and performs the fitting to the data"""

import vaex
import numpy as np
import matplotlib.pyplot as plt
from pysymmetry.visualization import grids, polarsubplot
import matplotlib
import lmfit as lm
from scipy.stats import moyal, norm
from scipy.integrate import trapezoid as trapz
# import pandas as pd
import pickle

#Selects hemisphere, circular variance and IMF Bz northward
hemi = 'north'
c_var_lim = 0.04
Bz_lim = 0

#IMF By selection, only select one at a time

#By_lim = 'data.By < -5'
#By_val = 'By < -5 nT'
#By_titl = '-5'

By_lim = 'data.By > 5'
By_val = 'By > 5 nT'
By_titl = '5'

#Tilt selection, only select one at a time

tilt_lim = 'data.tilt > 15'
tilt_titl = 15

#tilt_lim = 'data.tilt < -15'
#tilt_titl = -15


#Functions defining the distributions and the convolution integral
def gaussian(x, mean, sigma):
    # return 1/(np.sqrt(2*np.pi)*wid) * np.exp(-((x-cen)**2 / wid))
    return (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-mean)**2 / (2* sigma**2))

def moyal_(x,loc,scale):
    return moyal.pdf(x,loc,scale)
    
# Convolution integral, version with three parameters, change the variable locN to change the fixed offset value
def convolved_norm_moyal_pdf(x,
                             # locN=0.,scaleN=2.,
                             # locM=1.,scaleM=1.,
                             # mean=0.,
                             sigma=2.,
                             loc=1.,scale=1.,
                             # xvals=None,
):
    


    locN,scaleN = -150,sigma
    locM,scaleM = loc,scale


     # Make sure x is an array
    if not hasattr(x,"__len__"):
        x = [x]
    x = np.array(x)
    NZ = len(x)

    NX = 700
    xvals = np.linspace(-2000,5000,NX)
    xvals = np.broadcast_to(xvals,(NZ,NX))

    xnorm = (xvals-locN)/scaleN
    ynorm = (x[:,np.newaxis]-xvals)
    znorm = (ynorm-locM)/scaleM

    normvals = norm.pdf(xnorm)/scaleN
    moyvals = moyal.pdf(znorm)/scaleM
    
    return trapz(normvals*moyvals,dx=np.diff(xvals[0,:])[0],axis=1)

#Defines the function that fits the distribution and returns the Moyal mean based on the best fit parameters
def separation_function(arr):
    try:
        if len(arr) >= 10_000:

            y, bins = np.histogram(arr, bins=np.arange(-2000,5000, 10), density=True)
            x = np.array([0.5 * (bins[k] + bins[k+1]) for k in range(len(bins)-1)]) 
            
            result = mod.fit(y, fit_params, x=x, max_nfev=None)
            params =  result.best_values
            return moyal.mean(loc=params['loc'], scale=params['scale'])
        else:
            return np.nan
    except:pass
    
#Function to calculate the total mean based on the same data selection as the Moyal mean
def mean_(arr):
    try:
        if len(arr) >= 10_000:
            return np.mean(arr)
        else: return np.nan
    except:pass   
    
#List to conveniently select different files from the SSUSI dataset to perform analysis on
filnavn = '0100_0130'
fil_liste = [ 
                    # '0000', 
                    # '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
                    # '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', 
                    # '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', 
                    # '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', 
                    # '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050', 
                    # '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060', 
                    # '0061', '0062', '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070', 
                    # '0071', '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079', '0080',
                    # '0081', '0082', '0083', '0084', '0085', '0086', '0087', '0088', '0089', '0090', 
                    # '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', 
                    '0100',
                    '0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0110', 
                    '0111', '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', 
                    '0121', '0122', '0123', '0124', '0125', '0126', '0127', '0128', '0129', '0130', 
                    # '0131', '0132', '0133', '0134', '0135', '0136', '0137', '0138', '0139', '0140', 
                    # '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', 
                    # '0151', '0152', '0153', '0154', '0155', '0156', '0157', '0158', '0159', '0160', 
                    # '0161', '0162', '0163', '0164', '0165', '0166', '0167', '0168', '0169', '0170', 
                    # '0171', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', 
                    # '0181', '0182', '0183', '0184', '0185', '0186', '0187', '0188', '0189', '0190', 
                    # '0191', '0192'
            ]

#Path to the SSUSI data
datapath_work= '/mnt/7d7af4df-9a03-4816-8cab-8fb8579d9c1a/jone/Data/ssusi/vaex_hdf_merged/'
datapath = 'data/'

#Opens the omni data
omni = vaex.open(datapath+"omni.hdf5")

#Performs the data selection based on the IMF stability decided by bias vector length (Haaland et al. (2007)
window = 30
caStart = window # Minutes before each measurement
caEnd = 0 # Minutes after each measurement

omni['ca'] = omni.By.arctan2(omni.Bz)
omni['sin'] = np.sin(omni.ca.values)
omni['cos'] = np.cos(omni.ca.values)
omni = omni.to_pandas_df()
omni[['sinMean','cosMean']] = omni[['sin','cos']].rolling(window=1+caStart+caEnd, min_periods=0).mean()
omni = vaex.from_pandas(omni)
omni['caMean'] = omni.sinMean.arctan2(omni['cosMean'])
omni['c_var'] = 1-np.sqrt(omni['sinMean']**2+omni['cosMean']**2) # Circular variance

data = vaex.open_many([datapath_work+f"vaex_hdf_mergedchunked_{hemi}_{i}.hdf5" for i in fil_liste])

#Drops parts of the dataset not related to the current analysis
data = data[data['hemi'] == hemi]
data = data.drop('SP')
data = data.drop('SH')
data = data.drop('1304')
data = data.drop('1356')
data = data.drop('sat')
# data = data.drop('orbit')
data = data.drop('1216')
data = data.drop('lbhl')
data = data.drop('je')
data = data.drop('E0')
data = data.drop('datetime')
data = data.drop('hemi')

#Corrects day of year issue with our dataset
data.add_column('datetime', data.datetime_min.values - np.timedelta64(24,'h'))

#Joins omni and SSUSI dataset based on datetime with 1 min time resolution
data = data.join(omni, left_on=data.datetime, right_on=omni.index)
data = data.drop('index')
data = data.drop('datetime_min')


# data = data.drop('By')
# data = data.drop('Bz')
# data = data.drop('Vx')
data = data.drop('proton_density')

#Selects dipole tilt, and IMF conditions
data = data[eval(tilt_lim)]
data = data[data.Bz > Bz_lim]
data = data[eval(By_lim)]

#Selects circular variance
#-----------------------------------#
data = data[data.c_var < c_var_lim] #
#-----------------------------------#

data = data.drop('sin')
data = data.drop('cos')
data = data.drop('ca')
data = data.drop('cosMean')
data = data.drop('sinMean')
data = data.drop('caMean')

# data = data.drop('tilt') 
#Selects the restricted range to base analysis on, contains >99,7% of data
data = data[(data.lbhs >= -2000) & (data.lbhs <=5000)]

#Bins the lbhs-irradiances based on binnumber in our grid
group = 'binnumber'
with vaex.progress.tree('rich', title="Binning"):
    # pix_1 = data.lbhs.mean(binby='binnumber', shape=2413, progress=True)
    # count = data.lbhs.count(binby='binnumber', shape=2413, progress=True)
    binned = data.groupby(group, progress=True)

# pd_pix = pix_1.to_pandas_df(['binnumber','lbhs_mean'])
# pd_pix = pd_pix.reindex(index=pd.Series(np.arange(0,len(mltres))), method='nearest', tolerance=0.1)
# pix_1 = vaex.from_pandas(pd_pix)

#Sets up model to perform fitting
mod = lm.Model(convolved_norm_moyal_pdf)
fit_params = lm.Parameters()
# fit_params.add('mean', value=-101, min=-400, max=400)
fit_params.add('sigma', value=100, min=0, max=600)
fit_params.add('loc', value=100, min=-500, max=500)
fit_params.add('scale', value=100, min=0, max=600)

#Performs fitting for every bin above 60 MLAT
pix = list(map(pixelator, [binned.get_group([i]).lbhs.values for i in range(1030,2413)]))
pix_1 = list(map(mean_, [binned.get_group([i]).lbhs.values for i in range(1030,2413)]))
nans = [np.nan for i in range(0,1030)]

#Creates dataframe from fitting and exports it to a file
pix_df = vaex.from_arrays(binnumber = np.arange(0,2413),lbhs_pos = nans + pix, lbhs_pos_1 = nans +pix_1)
pix_df.export_hdf5(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_{c_var_lim}_{filnavn}.hdf5')

#Creates dictionary containing values of interest when comparing results
plot_info = {
    'orbits': np.unique(binned.get_group(2412).orbit.values).size,
    'avg_Bz': data.mean(data.Bz),
    'std_Bz': data.std(data.Bz),
    'avg_By': data.mean(data.By),
    'std_By': data.std(data.By),
    'avg_tilt': data.mean(data.tilt),
    'std_tilt': data.std(data.tilt),
    'date_start': data.head(n=1).datetime.values,
    'date_end': data.tail(n=1).datetime.values,
    'avg_Vx': data.mean(data.Vx),
    'std_Vx': data.std(data.Vx)
    }

#Pickles dictionary
with open(f'{hemi}/pdf/lmfit/{filnavn}/lmfit_tilt_{tilt_titl}_{By_titl}_cvar_{c_var_lim}_{filnavn}.pickle', 'wb') as handle:
    pickle.dump(plot_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
