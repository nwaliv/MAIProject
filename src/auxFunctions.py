import numpy as np
from yuv420 import readYUV420Range
from rgb2yuv_yuv2rgb import YUV2RGB
from skimage.util import view_as_windows
import scipy

def readFrames(inputVideo: str, resolution: tuple, t: int, numFrames: int):
    """
    Returns a selected tuple of frames from a YUV420 video, as RGB arrays
    """

    if t == 0:
        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t,t), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tmin1 = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t,t), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_t = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t+1,t+1), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tplus1 = YUV2RGB(YUVarr)
    elif t == numFrames - 1:
        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t-1,t-1), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tmin1 = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t,t), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_t = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t,t), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tplus1 = YUV2RGB(YUVarr)
    else:
        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t-1,t-1), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tmin1 = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t,t), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_t = YUV2RGB(YUVarr)

        Yarr, Uarr, Varr = readYUV420Range(inputVideo,resolution, (t+1,t+1), upsampleUV=True)
        YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
        YUVarr = np.moveaxis(YUVarr,0,-1)
        RGBarr_tplus1 = YUV2RGB(YUVarr)

    return RGBarr_tmin1,RGBarr_t,RGBarr_tplus1

def createOverlappingPatches(frame):
    """
    Returns a tuple of arrays of the input frame, as a rolling overlapping view of the original RGB array
    Note: Current function implementation will cause a reduction in frame size so pad the input arrays beforehand
    """
    Rarr = frame[:,:,0]; Garr = frame[:,:,1]; Barr = frame[:,:,2]
    Rwindow = view_as_windows(Rarr, (192,192),192) 
    Gwindow = view_as_windows(Garr, (192,192),192) 
    Bwindow = view_as_windows(Barr, (192,192),192)
    return Rwindow,Gwindow,Bwindow
    
def selectPatches(Rwindow,Gwindow,Bwindow,y:int,x:int):
    """
    Returns a patch from the rolling window view of the selected frame
    """
    Rpatch = Rwindow[y,x,:,:];Gpatch = Gwindow[y,x,:,:];Bpatch = Bwindow[y,x,:,:]
    Rpatch = np.expand_dims(Rpatch,axis=0); Gpatch = np.expand_dims(Gpatch,axis=0); Bpatch = np.expand_dims(Bpatch,axis=0)
    patch = np.concatenate((Rpatch, Gpatch, Bpatch), axis=0)
    patch = np.moveaxis(patch,0,-1)
    return patch

def image_preprocess(image):
    """
    Converts the image array from [0.0 ~ 255.0] -> [-1.0 ~ 1.0]
    """
    factor = 255.0 / 2.0
    center = 1.0
    image = image / factor - center  # [0.0 ~ 255.0] -> [-1.0 ~ 1.0]
    return image

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

def reconstruct(arr_actual, arr, patchSize=192, windowType='barthann'):
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
    reformed_shape = arr.shape 
    arr = np.reshape(arr, (-1, patchSize, patchSize, 3))

    if windowType == 'bartlett':
        window1d = scipy.signal.windows.bartlett(patchSize)
    elif windowType == 'blackman':
        window1d = scipy.signal.windows.bartlett(patchSize)
    if windowType == 'barthann':
        window1d = scipy.signal.windows.barthann(patchSize)
    window2d = np.reshape((np.outer(window1d, window1d)),(patchSize, patchSize, 1))
    arr_actual = arr_actual * window2d
    arr_actual = np.reshape(arr_actual, reformed_shape)
    outputArr = np.zeros((inputHeight+2*toPadY, inputWidth+2*toPadX, 3))
    for y in range(reformed_shape[0]):
        for x in range(reformed_shape[1]):
            currentSlice = np.s_[y*patchSize//2:(y*patchSize//2) + patchSize, x*patchSize//2:(x*patchSize//2) + patchSize, :]
            outputArr[currentSlice] = outputArr[currentSlice] + arr_actual[y, x, 0]
    outputArr = outputArr[toPadY:-toPadY, toPadX:-toPadX, :]
    return outputArr