#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:09:41 2023

@author: gaj002
"""

"""This script calculates solar zenith angle"""

import vaex
import numpy as np
# from pysymmetry.visualization import grids, polarsubplot
from pysymmetry.sunlight import sza
import matplotlib.pyplot as plt
import apexpy

hemi = 'north'
# filnavn = '0000_0030'
fil_liste = [
                    '0000', 
                    '0001', '0002','0003', 
                    '0004', '0005', '0006', '0007', 
                    '0008', '0009', '0010',
                    # '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', 
                    # '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', 
                    # '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', 
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
datapath_work= '/mnt/7d7af4df-9a03-4816-8cab-8fb8579d9c1a/jone/Data/ssusi/vaex_hdf_merged/'
datapath = 'data/'

omni = vaex.open(datapath+"omni.hdf5")

data = vaex.open_many([datapath_work+f"vaex_hdf_mergedchunked_{hemi}_{i}.hdf5" for i in fil_liste])

 try:    
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
    
    data.add_column('datetime_c', data.datetime.values - np.timedelta64(24,'h'))
    data_d = data[0]
    apex_out = apexpy.Apex(date=data_d[-1])
    # data['glat'] = data.apply(Apex.convert, arguments=[data.mlat,data.mlt, 'mlt', 'geo', data.datetime_c])
    print(i)
    glat_lon = apex_out.convert(lat=data.mlat.values, lon=data.mlt.values, source='mlt', dest='geo', datetime=data.datetime_c.values)
    # data['glat']= data.apply(apex_out.convert, arguments=[data.mlat,data.mlt, data.datetime_c])
    # data['glat'], data['glon'] = glat_lon[0], glat_lon[1]
    # data['tilt'] = data.apply(sza, arguments=[data.glat,data.glon, data.datetime_c], vectorize=True)
    
    sza_df = vaex.from_arrays(glat = glat_lon[0], glon = glat_lon[1], datetime_c=data.datetime_c.values)
    
    sza_df['sza'] = sza_df.apply(sza, arguments=[sza_df.glat,sza_df.glon, sza_df.datetime_c], vectorize=True)
    # except:pass
    sza_df.execute()
    # sza_df = sza_df.drop('glat')
    # sza_df = sza_df.drop('glon')
    # sza_df['sza'] = sza_df.sza.round()
    sza_df.export_hdf5(datapath+f"szangle/{hemi}_sza.hdf5",progress=True)
    print(f'{i} done')
except:pass
