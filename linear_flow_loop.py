# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:09:16 2024

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
from scipy import spatial as spt
from scipy import interpolate
import matplotlib.patches as mpatches
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import sys

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
halfBcs = int(np.floor(bcs/2))
bckChipSize = 9    ## this is the size of the chip used to identify the background
halfBkgChip = int(np.floor(bckChipSize/2))
diffAnomThresh = 11  ## this is how far above the background the center pixel must be to qualify as a starting point
dmSz = 155          ## the size of the sampling window
gaussSigma=1      ## size of the standard deviation of the gaussian smooting window
distThresh = 10
minCrackLength = 10

# %% gaussian filter of the hillshade or detrended topography
gaussHS = gaussian_filter(ddhs, sigma=gaussSigma)
gH = np.copy(gaussHS[0:dmSz,0:dmSz])
#gH = np.copy(ddhs[0:dmSz,0:dmSz])
mean_gH = np.mean(gH)

# %% ensure that there are only unique elements in the matrix
gH_U, uniq_idx, counts = np.unique(gH, return_index=True, return_inverse=True)
dims = np.shape(gH)
xDim = dims[1]
yDim = dims[0]
for xi in np.arange(xDim):
    for yj in np.arange(yDim):
        zVal = np.copy(gH[yj,xi])
        inds = np.where(gH==zVal)
        yval = np.squeeze(inds[0])
        xval = np.squeeze(inds[1]) 
        if np.size(yval)>1:
            boolT = np.logical_and((xval==xi),(yval==yj))
            yval = yval[boolT]
            xval = xval[boolT]
            gH[yval,xval]=np.copy(gH[yval,xval])+(np.random.default_rng().integers(low=1,high=9)*0.001)

# %% a chip is created that starts to scan across the window. when the center pixel of the chip is in sufficient contrast (large mean difference to the 
# average of the chip, we find a winner and start looking for a linear
lenLoop = 150   # this is how long you want to run before you find a mean difference that is acceptable
# mean_bckgChip = np.zeros(lenLoop)
# diffMeanM = np.zeros(lenLoop)
diffMean = 0
count = 0
crackElements = np.zeros((1,2))
gH_zeros = np.zeros(np.shape(gH))


for j in np.arange(halfBkgChip,dmSz-(halfBkgChip+1),2,dtype = int): # start the background chip search 1 in from the boundary so the small chip 
    for i in np.arange(halfBkgChip,dmSz-(halfBkgChip+1),dtype = int):
        
        cX = bckChipSize    # background chip size in X....choose an odd number
        cY = bckChipSize   # background chip size in Y
        intX_lf = int(i-halfBkgChip)   # left, right, up and down boundaries
        intX_rt = int(i+halfBkgChip+1)
        intY_up = int(j-halfBkgChip)
        intY_dn = int(j+halfBkgChip+1)
        
        bckgChip = np.copy(gH[intY_up:intY_dn,intX_lf:intX_rt])
        xCtr = int(i)
        yCtr = int(j)
        zCtr = np.copy(gH[yCtr,xCtr])    ## recheck that this is actually grabbing the center coordinate
        diffMean = zCtr - np.mean(bckgChip)
        if count==0:
            dCtr = 100
        if count>0:
            dCtr = np.sqrt(np.square(crackElements[:,0]-yCtr) + np.square(crackElements[:,1]-xCtr))
        # mean_bckgChip[i] = np.mean(bckgChip)
        # diffMeanM[i] = diffMean
        # if count==0:
        #     dCtr = 100
        # if count>0:
        #     dCtr = np.sqrt(np.square(crackElements[:,0]-yCtr) + np.square(crackElements[:,1]-xCtr))
        # instY = np.where(yCtr==expandedCrackULong[:,0])
        # instX = np.where(yCtr==expandedCrackULong[:,1])
        # inst = np.intersect1d(instX,instY)
        # NoCrackInChip = inst.size==0
        OneInBckChip = np.sum(gH_zeros[intY_up:intY_dn,intX_lf:intX_rt])
             
        if diffMean > diffAnomThresh and np.sum(dCtr<distThresh)==0:   # a center pixel stands out above the average and no other cracks 
                                                            # have been previously identified in other pixels in the background chip              
            
            print('you found a reasonable threshold')
            gH_cp = np.copy(gH)
            zMax = np.max(bckgChip)   # find where that chip has a max
            arB = gH==zMax
            ind_M = np.where(arB)
            #plt.imshow(chip_1,cmap = 'Grays')
            xPos = np.squeeze(ind_M[1]) 
            yPos = np.squeeze(ind_M[0]) 
            dCtr = np.sqrt(np.square(crackElements[:,0]-yPos) + np.square(crackElements[:,1]-xPos))
            
            k = 1
            maxMiddleY = np.copy(yPos)
            maxMiddleX = np.copy(xPos)
            while (diffMean > diffAnomThresh) and (xPos>=halfBkgChip and yPos>=halfBkgChip and yPos < (dmSz-halfBkgChip-1) and xPos < (dmSz-halfBkgChip-1)):
                # count += 1
                chip = np.copy(gH_cp[yPos-1:yPos+2,xPos-1:xPos+2])
                if k==1:
                    chip[0,:] = 0
                chip[1,1]=0
                gH_cp[yPos-1:yPos+2,xPos-1:xPos+2] = 0
                zMax = np.max(chip)   # find where that chip has a max
                if zMax ==0 :
                    break
                print(zMax)
                print(chip)
                arB = gH==zMax
                ind_M = np.where(arB)
                #plt.imshow(chip_1,cmap = 'Grays')
                xPos = np.squeeze(ind_M[1]) 
                yPos = np.squeeze(ind_M[0]) 
                maxMiddleY = np.vstack((maxMiddleY,np.copy(yPos)))
                maxMiddleX = np.vstack((maxMiddleX,np.copy(xPos)))
                
                intX_lf = int(xPos-halfBkgChip)   # left, right, up and down boundaries
                intX_rt = int(xPos+halfBkgChip+1)
                intY_up = int(yPos-halfBkgChip)
                intY_dn = int(yPos+halfBkgChip+1)
                bckgChip = np.copy(gH[intY_up:intY_dn,intX_lf:intX_rt])
                diffMean = zMax - np.mean(bckgChip)
                #print(diffMean)
                k+=1
         
            if maxMiddleY.size>minCrackLength:
                    
                 #plt.disableAutoRange()
                #plt.plot(maxMiddleX,maxMiddleY,'r-+')
                #plt.autoRange()
                crackElements = np.vstack((crackElements,np.hstack((maxMiddleY,maxMiddleX))))
                count+=1
                for m in np.arange(np.size(maxMiddleY)):
                    gH_zeros[maxMiddleY[m],maxMiddleX[m]] = 1
           
expandedCrackU = np.unique(crackElements,axis=0)
expandedCrackU = expandedCrackU[1:]
# %%

numEl = np.size(expandedCrackU[:,0])    # get the number of points in the expanded crack list
ordArr = np.copy(expandedCrackU[0,:])   # start a new 1x2 array that will be appended to with points in order
zeroArr= np.zeros(numEl)                # in this arrya, every time a point is reorder check it off the list by making it 1    
yMinEl = np.argmin(expandedCrackU[:,0])
yMin = np.min(expandedCrackU[:,0])
if np.size(yMinEl)>1:
    yList1 = np.copy(expandedCrackU)
    yList1[expandedCrackU[:,0]!=yMin,1]==1000
    xMinEl = np.argmin(yList1[:,1])
    k = xMinEl
else:
    k = yMinEl
crackList = []
innerCrackList = np.zeros((1,2))
count = 0

while np.sum(zeroArr==0)>1:
    zeroArr[k]=1
    yListLoop = np.copy(expandedCrackU[zeroArr==0,0])    # only grab a list of the x and y coords that have not been addded yet
    xListLoop = np.copy(expandedCrackU[zeroArr==0,1])
    x = expandedCrackU[k,1]                     # get the x and y indexed from the original loop
    y = expandedCrackU[k,0]
    innerCrackList = np.vstack((innerCrackList,np.hstack((y,x))))
    d2Els = np.squeeze(np.sqrt(np.square(expandedCrackU[:,0]-y) + np.square(expandedCrackU[:,1]-x)))  # find the distance between x and y and all other points in the original list
    d2Els0 = d2Els[zeroArr==0]    # get rid of the points that have already been accounted for
    minDist = min(d2Els0)
    if minDist > 2:
        if np.size(innerCrackList[:,0])>4:
            innerCrackList = innerCrackList[1:,:]
            crackList.append(innerCrackList)
        innerCrackList = np.zeros((1,2))

    minEl = np.where(d2Els0==np.min(d2Els0))
    count+=1
    if np.size(minEl)==2:
        k_inner = np.zeros(np.size(minEl))
        numNeigh = np.zeros(np.size(minEl)) 
        distNeigh = np.zeros(np.size(minEl)) 
        for m in np.arange(np.size(minEl)):
            in_minEl = np.copy(np.squeeze(minEl)[m])
            xNext = xListLoop[in_minEl]   # find the minimum 
            yNext = yListLoop[in_minEl]
            k_inner[m] = np.squeeze(np.where(np.logical_and((yNext==expandedCrackU[:,0]),(xNext==expandedCrackU[:,1]))))
            in_ind = int(k_inner[m])
            zeroArr[in_ind]=1
            # yListLoop = expandedCrackU[zeroArr==0,0]    # only grab a list of the x and y coords that have not been addded yet
            # xListLoop = expandedCrackU[zeroArr==0,1]
            # x = expandedCrackU[in_ind,1]                     # get the x and y indexed from the original loop
            # y = expandedCrackU[in_ind,0]
            d2Els = np.squeeze(np.sqrt(np.square(expandedCrackU[:,0]-yNext) + np.square(expandedCrackU[:,1]-xNext)))  # find the distance between x and y and all other points in the original list
            d2Els0 = d2Els[zeroArr==0]    # get rid of the points that have already been accounted for
            distNeigh[m] = np.min(d2Els0)
            numNeigh[m] = np.size(np.where(d2Els0==np.min(d2Els0)))
         
        if (np.sum(np.diff(numNeigh)) == 0) and (np.sum(np.diff(distNeigh)) == 0):
            print('theyre equal and equal')
            minEl_in = int(np.squeeze(minEl)[0])
        if (np.sum(np.diff(numNeigh)) == 0) and (np.sum(np.diff(distNeigh)) != 0):
            print('theyre equal')
            elMin_in = np.where(distNeigh == np.min(distNeigh))
            minEl_in = int(np.squeeze(minEl)[np.squeeze(elMin_in)])
        if (np.absolute(np.sum(np.diff(numNeigh))) > 0):
            print('theyre not equal')
            elMin_in = np.where(numNeigh == np.min(numNeigh))
            minEl_in = int(np.squeeze(minEl)[np.squeeze(elMin_in)])
        if minEl_in>np.size(xListLoop):
            print('too manu elements for xList lop')
            sys.exit()
        minEl = minEl_in        
        print('number of neighbors = ', numNeigh)
        print('finished the inner loop')
    if np.size(minEl)>2:
        minEl = np.squeeze(minEl)[0]
    print('outer loop')
    xNext = xListLoop[minEl]   # find the minimum 
    yNext = yListLoop[minEl]
    k = np.where(np.logical_and((yNext==expandedCrackU[:,0]),(xNext==expandedCrackU[:,1])))
    ordArr = np.vstack((ordArr,np.hstack((yNext,xNext))))

        
    
#plt.plot(ordArr[:,1],ordArr[:,0])   


for l in np.arange(len(crackList)):
    cracks = crackList[l] 
    #plt.plot(cracks[:,1],cracks[:,0],'b-')






# %%

# dNext = np.sqrt(np.square(ordArr[1:,0]-ordArr[0:-1,0])+np.square(ordArr[1:,1]-ordArr[0:-1,1]))
# # %% in this section of code you are taking all of the lines to 
# ##concatenate lines that are together and separating each line into 
# ##a list so that it becomes its own crack. doing this by finding the 
# ##distance between all pairs of points

# yy_T,yy = np.meshgrid(expandedCrackU[:,0],expandedCrackU[:,0])
# xx_T,xx = np.meshgrid(expandedCrackU[:,1],expandedCrackU[:,1])
# diff_X2 = np.square(xx_T-xx)
# diff_Y2 = np.square(yy_T-yy)

# bigDist = np.sqrt(2*np.square(dmSz))

# dist = np.sqrt(diff_X2 + diff_Y2)
# dist[dist==0]=bigDist

# crackCoordsOrd = np.array((yy_T[0,0],xx_T[0,0]))
# numEl = np.shape(xx_T)[0]
# lenLoop = numEl
# minDistM = np.zeros(lenLoop)
# beenCounted = np.ones(numEl)
# #for counter in np.arange(numEl):
# crackList = []
# innerCrackList = np.zeros((1,2))
# inCount = 0
    
# n_j = 0   # this is the column of the distance matrix or the elements in the expandedCrackU array
# for c_i in np.arange(numEl-1):
#     n_i = np.where(dist[np.squeeze(n_j),:]==np.min(dist[n_j,:]))   # n_i is the row where the colum n_j is at a minimum, once you fund it, go down to the n_i column, such that the new n_j = n_i
#     minDistM[c_i]=np.min(dist[n_j,:])
#     beenCounted[n_j]=0
#     newCoord = expandedCrackU[n_i]
#     crackCoordsOrd = np.vstack((crackCoordsOrd,newCoord))
#     innerCrackList = np.vstack((innerCrackList,newCoord))
#     dist[n_i,n_j]=bigDist
#     print('n_j, n_i =',n_j,np.squeeze(n_i))
#     n_j = np.copy(np.squeeze(n_i))
    
#     print('c_i=',c_i)
#     print('minDist =',np.squeeze(minDistM[c_i]))
#     print('sum of beenCounted =', np.sum(beenCounted))
    
#     # if c_i == 272:
#     #     sys.exit()
    
#     if minDistM[c_i] >= 2:
#         innerCrackList = innerCrackList[1:,:]
#         crackList.append(innerCrackList) 
#         innerCrackList = np.zeros((1,2))
#         stillInY = np.multiply(beenCounted,expandedCrackU[:,0])
#         n_j = np.where(stillInY==np.min(stillInY[stillInY!=0]))
#         inCount+=1
#         # if inCount == 4:
#         #     print('inCount =4')
#         #     #sys.exit()
#         if np.size(n_j)>1:
#             print('houston we have a problem')
#             sys.exit()
#             n_j = n_j[0]

# innerCrackList = innerCrackList[1:,:]
# crackList.append(innerCrackList)

# %% Here you are sorting the cracks, then 
crackListAve = []
crackListFit = []
numCracks = len(crackList)
r2 = np.zeros(numCracks)

for crk in np.arange(numCracks):
# crk = 3
    crack = crackList[crk]
    crack_s = np.sort(crack,axis=0)
    c_x = crack[:,1]
    c_y = crack[:,0]
    #plt.plot(crack[:,1],crack[:,0],'b-')
    #plt.plot(c_x,c_y,'+')
    
    c_xU = np.unique(c_x)
    sortEls = np.argsort(c_xU)
    c_xU = np.sort(c_xU)
    lx = np.size(c_xU)
    meanY = np.zeros(lx)
    for r in np.arange(lx):
        elX = np.where(c_x==c_xU[r])
        meanY[r] = np.mean(c_y[elX]) 
    
    crackListAve.append(np.rot90(np.vstack((c_xU,meanY)),3))
    #plt.plot(c_xU,meanY,'o')
    xnew = np.arange(c_xU[0],c_xU[-1]+1)
    cspl= interpolate.CubicSpline(c_xU,meanY)
    ynew = cspl(c_xU)
    #plt.plot(xnew,ynew,'r-')
    pfit = np.polyfit(c_xU,meanY,4)
    p = np.poly1d(pfit)
    pFit = p(c_xU)
    
    crackListFit.append(np.rot90(np.vstack((c_xU,pFit)),3))
    
    mY = np.mean(c_y)
    SSres = np.zeros(lx)
    SStot = np.zeros(lx)
    for r in np.arange(lx):
        elX = np.where(c_x==c_xU[r])
        SSres[r] = np.sum(np.square(c_y[elX]-pFit[r]))   
        SStot[r] = np.sum(np.square(c_y[elX]-mY))
        
    SSresT = np.sum(SSres)
    SStotT = np.sum(SStot)
    r2[crk] = 1 - np.divide(SSresT,SStotT)
    
    if r2[crk]>0.8:
        plt.plot(c_xU,pFit,'go')

#%%
#plt.plot(np.flipud(expandedCrackU[:,1]),np.flipud(expandedCrackU[:,0]),'.')
