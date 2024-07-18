# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:25:26 2024

@author: jcrompto
"""

from osgeo import gdal
import matplotlib.pyplot as plt


# %% 
fileName = r'C:\Users\jcrompto\Documents\remote_sensing\lidar\joffre\LiDAR Raster\20_4031_01_1m_HS_CSRS_z10.tif'
dataSet = gdal.Open(fileName,gdal.GA_ReadOnly)
band = dataSet.GetRasterBand(1)
arr = band.ReadAsArray()
plt.imshow(arr)

# %%

hS = gdal.gdaldem

