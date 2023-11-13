#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:47:07 2023

@author: gaj002
"""
"""This script plots intensity vs solar zenith angle """

import vaex
import numpy as np
# from pysymmetry.visualization import grids, polarsubplot
from pysymmetry.sunlight import sza
import matplotlib.pyplot as plt
import apexpy

hemi = 'north'
filnavn = '0031_0040'
mlat = '60_80'
fil_liste = [ 
                    # '0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
                    # '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', 
                    # '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', 
                    '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', 
                    # '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050', 
                    # '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060', 
                    # '0061', '0062', '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070', 
                    # '0071', '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079', '0080',
                    # '0081', '0082', '0083', '0084', '0085', '0086', '0087', '0088', '0089', '0090', 
                    # '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', '0100',
                    # '0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0110', 
                    # '0111', '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', 
                    # '0121', '0122', '0123', '0124', '0125', '0126', '0127', '0128', '0129', '0130', 
                    # '0131', '0132', '0133', '0134', '0135', '0136', '0137', '0138', '0139', '0140', 
                    # '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', 
                    # '0151', '0152', '0153', '0154', '0155', '0156', '0157', '0158', '0159', '0160', 
                    # '0161', '0162', '0163', '0164', '0165', '0166', '0167', '0168', '0169', '0170', 
                    # '0171', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', 
                    # '0181', '0182', '0183', '0184', '0185', '0186', '0187', '0188', '0189', '0190', 
                    # '0191', 
                    # '0192'
            ]
# mlt_start, mlt_end = 12,18
datapath_work= '/mnt/7d7af4df-9a03-4816-8cab-8fb8579d9c1a/jone/Data/ssusi/vaex_hdf_merged/'
datapath = 'data/'

data = vaex.open_many([datapath_work+f"vaex_hdf_mergedchunked_{hemi}_{i}.hdf5" for i in fil_liste])
angle = vaex.open_many([datapath_work+f"szangle/{hemi}_sza_{i}.hdf5" for i in fil_liste])
data.add_column('datetime_co', data.datetime.values - np.timedelta64(24,'h'))
data = data.join(angle)
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
# data.add_column('datetime_c', data.datetime.values - np.timedelta64(24,'h'))
data = data[(data.lbhs >= -10_000) & (data.lbhs <=10_000)]
data = data[(data.mlat >= 60) & (data.mlat <= 80)]
# data = data[data.mlat <= mlat]
# data = data[(data.mlt >= mlt_start) & (data.mlt <=mlt_end)]

mlt_lst = [
            [0,6],
            [6,12],
            [12,18],
            [18,24]
            ]


data_1 = data[(data.mlt >= mlt_lst[0][0]) & (data.mlt <=mlt_lst[0][1])]
data_2 = data[(data.mlt >= mlt_lst[1][0]) & (data.mlt <=mlt_lst[1][1])]
data_3 = data[(data.mlt >= mlt_lst[2][0]) & (data.mlt <=mlt_lst[2][1])]
data_4 = data[(data.mlt >= mlt_lst[3][0]) & (data.mlt <=mlt_lst[3][1])]

data_5 = data.copy()
corr = data_5.correlation('sza','lbhs')

binned_mean_1 = data_1.mean(data_1.lbhs, binby = data.sza, shape=80)
binned_std_1 = data_1.std(data_1.lbhs, binby = data.sza, shape=80)
binned_mean_2 = data_2.mean(data_2.lbhs, binby = data.sza, shape=80)
binned_std_2 = data_2.std(data_2.lbhs, binby = data.sza, shape=80)
binned_mean_3 = data_3.mean(data_3.lbhs, binby = data.sza, shape=80)
binned_std_3 = data_3.std(data_4.lbhs, binby = data.sza, shape=80)
binned_mean_4 = data_4.mean(data_4.lbhs, binby = data.sza, shape=80)
binned_std_4 = data_4.std(data_4.lbhs, binby = data.sza, shape=80)

binned_mean_5 = data_5.mean(data_5.lbhs, binby = data.sza, shape=80)
binned_std_5 = data_5.std(data_5.lbhs, binby = data.sza, shape=80)

fig = plt.figure(figsize=(10,6))
# plt.scatter(data.sza.values, data.lbhs.values, s=0.01, alpha=0.25)
plt.errorbar(range(50,130), binned_mean_1, yerr=binned_std_1, alpha=0.5, label=f"{mlt_lst[0][0]}_{mlt_lst[0][1]}")
plt.errorbar(range(50,130), binned_mean_2, yerr=binned_std_2, alpha=0.5, label=f"{mlt_lst[1][0]}_{mlt_lst[1][1]}")
plt.errorbar(range(50,130), binned_mean_3, yerr=binned_std_3, alpha=0.5, label=f"{mlt_lst[2][0]}_{mlt_lst[2][1]}")
plt.errorbar(range(50,130), binned_mean_4, yerr=binned_std_4, alpha=0.5, label=f"{mlt_lst[3][0]}_{mlt_lst[3][1]}")
plt.errorbar(range(50,130), binned_mean_5, yerr=binned_std_5, alpha=1, linewidth = 2,label="tot")


plt.xlabel('Solar zenith angle')
plt.ylabel('Intensity')
plt.title(f'mean_std sza, intensity, mlat, {mlat}, {filnavn}')
plt.legend()
plt.savefig(f'scatter_solar/binned_mean_mlat_{mlat}_{filnavn}.pdf')

