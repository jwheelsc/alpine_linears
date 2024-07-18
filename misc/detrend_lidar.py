# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:43:29 2024

@author: jcrompto
"""

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio.plot 
import rasterio as rio
import xdem
from skimage.morphology import erosion, dilation, opening, closing, white_tophat  
from skimage.morphology import black_tophat, skeletonize, convex_hull_image 
from skimage.morphology import disk, square, diamond
from skimage import filters, feature
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from scipy.interpolate import RegularGridInterpolator

filename = r'C:\Users\jcrompto\Documents\remote_sensing\lidar\joffre\LiDAR Raster\20_4031_01_1m_DTM_CSRS_z10_ellips.tif'
rast = gu.Raster(filename)

llx = 536802; lly = 5578395; urx = 537774; ury = 5579407;
rast.crop((llx,lly,urx,ury),inplace = True)
rast.show(ax = "new",cmap = "Greys_r")

# %% 

rastData = rast.data
t = np.linspace(np.min(rastData),np.max(rastData),983664)

xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
y = np.linspace(1,ys,ys)
xg,yg = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(xg.ravel(), yg.ravel(), rastData.ravel(), c = rastData, cmap = "Grays", s=80, label='data')
# %%
ax.view_init(40,60)
plt.show()

# %%
xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
y = np.linspace(1,ys,ys)
xg,yg = np.meshgrid(x,y)

interpDat = RegularGridInterpolator((x,y),rastData,method='linear')