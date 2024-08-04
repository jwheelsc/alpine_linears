# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:43:10 2024

@author: jcrompto
"""
# %% import raster and plotting tools

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio.plot 
import rasterio as rio
import xdem
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as mpatches
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import sys

# %%

filename = r'C:\Users\jcrompto\Documents\remote_sensing\lidar\joffre\LiDAR Raster\20_4031_01_1m_DTM_CSRS_z10_ellips.tif'
rast = gu.Raster(filename)
left = rast.bounds[0]
right = rast.bounds[2]
bottom = rast.bounds[1]
top = rast.bounds[3]
rastDat = rast.data
hs,slope = hillshade(rastDat,270,180)

fig, ax = plt.subplots(1)
ax.imshow(hs,extent = [left, right, bottom, top] , cmap = "Greys_r")

# %%

finalCrackList = []

for jj in np.arange(21):
    for ii in np.arange(21):
        
        openFold = 'C:\\Users\\jcrompto\\Documents\\code\\python_scripts\\detect_alpine_linears\\saveCracks\\Joffre\\'
        strCrk = openFold + 'keepCracks_' + str(jj) + '_' + str(ii) + '.npz'
        strCrk_stats = openFold + 'keepCracks_' + str(jj) + '_' + str(ii) + '_stats.npz'
        npz = np.load(strCrk, allow_pickle=True)
        xCoords=npz['arr_0']
        yCoords=npz['arr_1']
        npz_stats = np.load(strCrk_stats, allow_pickle=True)
        slpCrack = npz_stats['arr_0']
        r2Crack = npz_stats['arr_1']
        chordCrack = npz_stats['arr_2']
        
        if np.size(xCoords)>1:
            for p in np.arange(len(xCoords)):
                x = xCoords[p]
                y = yCoords[p]
                setList = np.hstack((y,x))
                finalCrackList.append(setList)
                ax.plot(np.flipud(y),np.flipud(x),'.')
                
        
    
# %%
def hillshade(array,azimuth,angle_altitude):
    azimuth = 360 - azimuth
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return (255*(shaded + 1)/2, slope)
