import numpy as np
from yuv420 import readYUV420Range
from rgb2yuv_yuv2rgb import YUV2RGB
from skimage.util import view_as_windows

def deconstruct(arr, patchSize=192):
    inputHeight, inputWidth = arr.shape[0], arr.shape[1]

    #PAD HEIGHT AND WIDTH TO MULTIPLE OF PATCHSIZE
    nearestHeight = 0 
    nearestWidth = 0 
    cnt = 0 
    while nearestWidth < inputWidth:
        nearestWidth = cnt * patchSize
        cnt += 1 
    cnt = 0 
    while nearestHeight < inputHeight:
        nearestHeight = cnt * patchSize
        cnt += 1 
    toPadX, toPadY = ((nearestWidth - inputWidth + patchSize)//2), (nearestHeight - inputHeight + patchSize//2)
    arr = np.pad(arr, ((toPadY, toPadY),(toPadX, toPadX), (0,0)), 'edge')
    arr = view_as_windows(arr, (patchSize, patchSize, 3), step=patchSize//2)
    arr = np.reshape(arr, (-1, patchSize, patchSize, 3))
    # Variable "arr" are N * patchSize * patchSize * 3 patches with some overlap 
    return arr
