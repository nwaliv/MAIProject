import numpy as np
from yuv420 import readYUV420Range
from rgb2yuv_yuv2rgb import YUV2RGB
from skimage.util import view_as_windows

def readFrames(input_video: str, resolution: tuple, t: int, numFrames: int):
    """
    Returns a selected tuple of frames from a YUV420 video, as RGB arrays
    """
    Yarr, Uarr, Varr = readYUV420Range(input_video,resolution, (t-1,t-1), upsampleUV=True)
    YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
    YUVarr = np.moveaxis(YUVarr,0,-1)
    RGBarr_tmin1 = YUV2RGB(YUVarr)

    Yarr, Uarr, Varr = readYUV420Range(input_video,resolution, (t,t), upsampleUV=True)
    YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
    YUVarr = np.moveaxis(YUVarr,0,-1)
    RGBarr_t = YUV2RGB(YUVarr)

    Yarr, Uarr, Varr = readYUV420Range(input_video,resolution, (t+1,t+1), upsampleUV=True)
    YUVarr = np.concatenate((Yarr,Uarr,Varr), axis=0)
    YUVarr = np.moveaxis(YUVarr,0,-1)
    RGBarr_tplus1 = YUV2RGB(YUVarr)

    if t == 0:
        return RGBarr_t,RGBarr_t,RGBarr_tplus1
    elif t == numFrames - 1:
        return RGBarr_tmin1,RGBarr_t,RGBarr_t
    else:
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