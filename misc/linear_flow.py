# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:15:23 2024

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
from scipy import signal
import matplotlib.patches as mpatches

# %% import skimage tools
from skimage.morphology import erosion, dilation, opening, closing, white_tophat  
from skimage.morphology import black_tophat, skeletonize, convex_hull_image 
from skimage.morphology import disk, square, diamond
from skimage import filters, feature
from skimage.filters import threshold_otsu
from skimage.filters.rank import median
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# %% import scipy tools
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

# %% import the the raster files created by another program
dd_rast = gu.Raster(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM.tif')
ddhs_rast = gu.Raster(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM_hs.tif')


# %% plot the raster data being imorted
ddhs = ddhs_rast.data
dd = dd_rast.data
plt.imshow(ddhs,cmap = 'Grays')

# %% set up some paramters
bcs = 5             ## this is the size of the center chip used to move through the linear trough
bckChipSize = 9     ## this is the size of the chip used to identify the background
diffAnomThresh = 25 ## this is how far above the background the center pixel must be to qualify as a starting point
dmSz = 150          ## the size of the sampling window
gaussSigma=1        ## size of the standard deviation of the gaussian smooting window

# %% gaussian filter of the hillshade or detrended topography
gaussHS = gaussian_filter(ddhs, sigma=gaussSigma)
gH = np.copy(gaussHS[0:dmSz,0:dmSz])
gH_cp = np.copy(gH)
mean_gH = 


# %% a chip is created that starts to scan across the window. when the center pixel of the chip is in sufficient contrast (large mean difference to the 
# average of the chip, we find a winner and start looking for a linear
lenLoop = 100   # this is how long you want to run before you find a mean difference that is acceptable
mean_bckgChip = np.zeros(lenLoop)
ctrZ =  np.zeros(lenLoop)
diffMeanM = np.zeros(lenLoop)
diffMean = 0
i = 0
j = 0

while diffMean < diffAnomThresh:

    cX = bckChipSize    # background chip size in X....choose an odd number
    cY = np.copy(cX)   # background chip size in Y
    intX_lf = i   # left, right, up and down boundaries
    intX_rt = intX_lf+cX
    intY_up = j
    intY_dn = intY_up+cY
    cX_c = np.floor(cX/2)

    bckgChip = gH_cp[intY_up:intY_dn,intX_lf:intX_rt]
    xCtr = int(intX_lf + (cX_c))
    yCtr = int(intY_up + (cX_c))
    ctr = np.copy(gH_cp[yCtr,xCtr])    ## recheck that this is actually grabbing the center coordinate

    mean_bckgChip[i] = np.mean(bckgChip)
    ctrZ[i] = ctr
    diffMean = ctr - np.mean(bckgChip)
    diffMeanM[i] = ctr - np.mean(bckgChip)

    i += 1

plt.plot(mean_bckgChip)
# plt.plot(mean_bckgChip_ctr)
plt.plot(ctrZ)
plt.plot(diffMeanM)

# %% once you have found a qualifying pixel, make a larger chip in it's neighbourhood and find the max value and correspoding coordinates
chip_1  = np.copy(gH_cp[yCtr-2:yCtr+3,xCtr-2:xCtr+3])
zMax = np.max(chip_1)   
arB = gH_cp==zMax
ind_1 = np.where(arB)
plt.imshow(chip_1,cmap = 'Grays')

xPos = np.squeeze(ind_1[1]) 
yPos = np.squeeze(ind_1[0]) 
chipN = np.copy(gH_cp[yPos-1:yPos+2,xPos-1:xPos+2])
plt.imshow(chipN,cmap = 'Grays')
print(chipN)

# %% now that you;ve found the max in the neighbourhood of the qualifying pixel, start with that as the center and carve through the feature

chip = np.copy(chipN)
chip[1,1]=0
maxMiddleY = yPos
maxMiddleX = xPos
#print('first chip = ', chip)

#mumIts = 40
#bcs = 5
diffAnomM = zMax/2
diffAnom = zMax/2


k = 1
while diffAnom > diffAnomThresh:
    if k == 1:
        chip[0,:]=0
    zMax = np.max(chip)
    arB = gH==zMax
    indNext = np.where(arB)
    gH[yPos-1:yPos+2,xPos-1:xPos+2] = 0
    if yPos<=(bcs+1):
        gHBig = gH_cp[yPos:yPos+(2*bcs)+1,xPos-bcs:xPos+bcs+1]
        print('its less')
    elif yPos>bcs:
        gHBig = gH_cp[yPos-bcs:yPos+bcs+1,xPos-bcs:xPos+bcs+1]
        print('its bigger')
    meanBIG = np.mean(gHBig)
    # if np.isnan(meanBIG):
    #     break
    print('meanBIG = ', meanBIG)
    
    yPos =  np.squeeze(indNext[0])
    xPos  = np.squeeze(indNext[1]) 
    
    #plt.imshow(gH)
    maxMiddleY = np.hstack((maxMiddleY,yPos))
    maxMiddleX = np.hstack((maxMiddleX,xPos))
    chip = np.copy(gH[yPos-1:yPos+2,xPos-1:xPos+2])
    diffAnom = zMax - meanBIG
    print(diffAnom)
    diffAnomM = np.hstack((diffAnomM,diffAnom))
    k +=1
    print(chip)
    

plt.imshow(gH,cmap = 'Grays')
plt.plot(maxMiddleX,maxMiddleY)




