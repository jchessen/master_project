#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:16:50 2023

@author: gaj002
"""

"""This script filters the data according to dipole tilt angle performs the fitting to the data to examine the performance of the fit to the distributons of the data"""

import vaex
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
from scipy.integrate import trapezoid as trapz
from scipy.stats import moyal, norm

#select hemisphere and tilt
hemi = 'north'
tilt = 15

#Defines functions
def gaussian(x, mean, sigma):
    return (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-mean)**2 / (2* sigma**2))
def moyal_(x,loc,scale):
    return moyal.pdf(x,loc,scale)

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

    
#List to conveniently select different files from the SSUSI dataset to perform analysis on
filnavn = '0000_0030'
fil_liste = [
                    '0000', 
                    '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
                    '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', 
                    '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', 
                    # '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', 
                    # '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050', 
                    # '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060', 
                    # '0061', '0062', '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070', 
                    # '0071', '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079', '0080',
                    # '0081', '0082', '0083', '0084', '0085', '0086', '0087', '0088', '0089', '0090', 
                    # '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', 
                    # '0100',
                    # '0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0110', 
                    # '0111', '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', 
                    # '0121', '0122', '0123', '0124', '0125', '0126', '0127', '0128', '0129', '0130', 
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
data = vaex.open_many([datapath_work+f"vaex_hdf_mergedchunked_{hemi}_{i}.hdf5" for i in fil_liste])
#Drops parts of the dataset not related to the current analysis
data = data[data['hemi'] == hemi]
data = data.drop('SP')
data = data.drop('SH')
data = data.drop('1304')
data = data.drop('1356')
data = data.drop('sat')
data = data.drop('orbit')
data = data.drop('1216')
data = data.drop('lbhl')
data = data.drop('je')
data = data.drop('E0')
data = data.drop('datetime')

#Corrects day of year issue with our dataset
data.add_column('datetime', data.datetime_min.values - np.timedelta64(24,'h'))

#Joins omni and SSUSI dataset based on datetime with 1 min time resolution
data = data.join(omni, left_on=data.datetime, right_on=omni.index)
data = data.drop('index')
data = data.drop('datetime_min')
data = data.drop('By')
data = data.drop('Bz')
data = data.drop('Vx')
data = data.drop('proton_density')
data = data[data.tilt > tilt]

#Selects the restricted range to base analysis on, contains >99,7% of data
data = data[(data.lbhs >= -2000) & (data.lbhs <=5000)]


#Exclude or include spike
# data = data[(data.lbhs >=-50) | (data.lbhs <=-60)]


#Bins the lbhs-irradiances based on binnumber in our grid
group = 'binnumber'
with vaex.progress.tree('rich', title="Binning"):
    binned = data.groupby(group, progress=True)
    
#Sets up model to perform fitting
mod = lm.Model(convolved_norm_moyal_pdf)
fit_params = lm.Parameters()
# fit_params.add('mean',value=-101, min=-400,max=400)
fit_params.add('sigma', value=100, min=0, max=400)
fit_params.add('loc', value=100, min=-500, max=500)
fit_params.add('scale', value=100, min=0, max=800)

#Performs fit to every bin and saves a histogram with plotted distributions
for i in range(0,2413):
    try:
        intens1 = binned.get_group([i]).lbhs.values
        if len(intens1) >= 1_000:
            fig, ax1 = plt.subplots(1,1, figsize=(6,8))

            title = f'mlat: {binned.get_group([i]).head(n=1).mlat.values}, mlt: {binned.get_group([i]).head(n=1).mlt.values}'
            y, bins = np.histogram(intens1, bins=np.arange(-2000,5000, 10), density=True)
            x = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)]) 
            
            result1 = mod.fit(y, fit_params, x=x)

            params1 = result1.best_values
            ax1.hist(intens1, bins=700, range=[-2000,5000], label='Data', density=True)
            
            ax1.plot(x, result.init_fit, '--', label='initial fit')
            ax1.plot(x, result1.best_fit, '-', label='Best fit')
            ax1.plot(x, moyal_(x, params1['loc'], params1['scale']), label='Moyal')
            ax1.plot(x, gaussian(x, -150, params1['sigma']), label='Gauss')
            
            ax1.text(0.7,0.25,'Moyal mean:' + f"{round(moyal.mean(loc=params1['loc'], scale=params1['scale']),2)}" , transform = ax1.transAxes, size=6)
            ax1.text(0.7,0.2, 'Mean:' +  f"{np.format_float_positional(np.mean(intens1), 2)}" , transform = ax1.transAxes, size=6)
            ax1.text(0.7,0.15,'Moyal median:' + f"{round(moyal.median(loc=params1['loc'], scale=params1['scale']),2)}" , transform = ax1.transAxes, size=6)
            ax1.text(0.7,0.1, 'Median:' +  f"{np.format_float_positional(np.median(intens1),2)}" , transform = ax1.transAxes, size=6)

            ax1.legend()
            plt.xlabel('Intensity [R]')
            
            plt.tight_layout()
            plt.savefig(f"fitted_hist/hist_{filnavn}_{i}.pdf")
            plt.show()
            fig.clf()
            plt.close()
            print(result1.fit_report())
    except:pass

