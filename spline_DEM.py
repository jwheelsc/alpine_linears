# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:11:32 2024

@author: jcrompto
"""

#!/usr/bin/env python
# coding: utf-8

# in this script, lidar is imported, gridded, an interpolated with a spline
# In[210]:

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio.plot 
import rasterio as rio
import xdem
from scipy.interpolate import RegularGridInterpolator


from skimage.morphology import erosion, dilation, opening, closing, white_tophat  
from skimage.morphology import black_tophat, skeletonize, convex_hull_image 
from skimage.morphology import disk, square, diamond
from skimage import filters, feature, exposure, segmentation
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
# %%
filename = r'C:\Users\jcrompto\Documents\remote_sensing\lidar\joffre\LiDAR Raster\20_4031_01_1m_DTM_CSRS_z10_ellips.tif'
rast = gu.Raster(filename)

# In[]

# llx = 535000; lly = 5574700; urx = 540300; ury = 5580000
#llx = 536802; lly = 5578395; urx = 537774; ury = 5579407;
#llx = 537000; lly = 5579000; urx = 537200; ury = 5579200; #this is typically the original
#llx = 537000-2000; lly = 5579000-2000; urx = 537200-2000; ury = 5579200-2000;  #this is typically the _2
#llx = 536500; lly = 5578500; urx = 537900; ury = 5579900;  #this is a bigger domain


LLX = 535000; LLY = 5574700; URX = 540300; URY = 5580000
distX = URX-LLX
distY = URY-LLY
overlap = 50
numBlock = 21
xDom = np.divide(distX-overlap,numBlock) + overlap
yDom = np.divide(distY-overlap,numBlock) + overlap

leftBounds = np.linspace(LLX,URX-xDom,21)
rightBounds = np.linspace(LLX+xDom-overlap,URX,21)
lowerBounds = np.linspace(LLY,URY-yDom,21)
upperBounds = np.linspace(LLY+yDom-overlap,URY,21)

for jj in np.arange(np.size(upperBounds)):
    ury = upperBounds[jj]
    lly = lowerBounds[jj]
    for ii in np.arange(np.size(rightBounds)):
        llx = leftBounds[ii]
        urx = rightBounds[ii]
        rast.crop((llx,lly,urx,ury),inplace = True)
        rast.show(ax = "new",cmap = "Greys_r")


# here you are simply plotting the data above but in 3D view and downsampled by dx

# In[197]:

rastData = rast.data
t = np.linspace(np.min(rastData),np.max(rastData),983664)

xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
y = np.linspace(ys,1,ys)
xg,yg = np.meshgrid(x,y)

dx = 20

xgds = xg[::dx,::dx]
ygds = yg[::dx,::dx]
rdds = rastData[::dx,::dx]

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
#ax.scatter(xgds.ravel(), ygds.ravel(), rdds.ravel(), c = rdds, cmap = "Grays", s=60,  label='data', marker = '.')
#ax.scatter(xgds, ygds, rdds, c = rdds, cmap = "Grays", s=60,  label='data', marker = '.',alpha=0.4)

ax.plot_wireframe(np.rot90(xgds),np.rot90(ygds),(rdds), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
#ax.view_init(-40,60)


# in this cell, I am using an interpolator on the downscaled data, then using the smnoothed interpolation to resample at the original resolution

# In[198]:


xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
dx = 20
xds = x[::dx]
y = np.linspace(ys,1,ys)
yds = y[::dx]
xgds,ygds = np.meshgrid(xds,yds)
rdds = rastData[::dx,::dx]
interpDat = RegularGridInterpolator((xds,yds),rdds,method='cubic',bounds_error=False,)
int1m = interpDat((xg,yg))
dxCubic = np.fliplr(interpDat.values)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(np.rot90(xgds),np.rot90(ygds),interpDat.values, rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
#ax.plot_wireframe(np.rot90(xg),np.rot90(yg),(interpDat.values), rstride=3, cstride=3,
#                  alpha=0.8, color='b', label='linear interp')
#ax.view_init(-40,60)


# In[214]:


intCubic = np.fliplr(np.rot90(int1m,-1))
diffDEM = rastData-intCubic
hs,slope = hillshade(diffDEM,270,180)


# In[215]:


fig, (ax) = plt.subplots(1,4)
ax[0].imshow(dxCubic,cmap = 'jet',extent = [0,xs,0,ys])
ax[0].set_title('20m spline')
ax[1].imshow(rastData,cmap = 'jet')
ax[1].set_title('data')
ax[2].imshow(intCubic,cmap = 'jet',extent = [0,xs,0,ys])
ax[2].set_title('resampled from interpolated')
ax[3].imshow(diffDEM,cmap = 'jet',extent = [0,xs,0,ys])
ax[3].set_title('diff DEM')


# %% here you are saving all of the raster data


dd_rast = rast.copy(new_array=diffDEM)
dd_rast.save(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM_big.tif')


ddhs_rast = rast.copy(new_array=hs)
ddhs_rast.save(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM_hs_big.tif')

DEM_rast = rast.copy(new_array=rastData)
DEM_rast.save(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\rastDEM_big.tif')

# In[207]:


def hillshade(array,azimuth,angle_altitude):
    azimuth = 360 - azimuth
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return (255*(shaded + 1)/2, slope)


# In[ ]:


def double_gradient(array):
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    x2, y2 = np.gradient(slope)
    del_slope = np.pi/2. - np.arctan(np.sqrt(x2*x2 + y2*y2))
    
    return del_slope

