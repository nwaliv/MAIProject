
from library import readYUV420, YUV2RGB, hrNet
from library.GeneralOps import RGB2YUV, writeYUV420, readYUV2RGB
from src.auxFunctions import deconstruct, reconstruct
import numpy as np
import tensorflow as tf
import os
import time



# baseCompPath = '/home/nwaliv/bandon/nwaliv/testVideoSetDeg/'
# vidName = "wipe_3840x2160_5994_10bit_420_6309-6439_AV1_CRF26_.yuv"
# _comp = os.path.join(baseCompPath,vidName)
# _width = 1920
# _height = 1080
# _numFrames = 130
# _numFramesRestored = 10
# #outVidName = "output/res_Dolby.yuv"
# #modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
# BATCHSIZE = 16

#Testing for 1080p Video
baseCompPath = '/data/nwaliv/trainVideoSetDeg/'
vidName = "DOLBY_ATMOS_UNFOLD_2_FEEL_EVERY_DIMENSION_LOSSLESS-thedigitaltheater_4_AV1_CRF38_.yuv"
_comp = os.path.join(baseCompPath,vidName)
_width = 1920
_height = 1080
_numFrames = 120
_numFramesRestored = 10
outVidName = "output/res_Dolby.yuv"
modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
BATCHSIZE = 4


# Works for 480p video
# baseCompPath = '/data/nwaliv/trainVideoSetDeg/'
# vidName = "DCloudsStaticBVITexture_480x272_120fps_10bit_420_H264_CRF31_.yuv"
# _comp = os.path.join(baseCompPath,vidName)
# _width = 480
# _height = 272
# _numFrames = 64
# outVidName = "output/res_DClouds.yuv"
# modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
# BATCHSIZE = 16

# baseCompPath = 'dataset/'
# vidName = "canvas_480x272.yuv"
# _comp = os.path.join(baseCompPath,vidName)
# _width = 480
# _height = 272
# _numFrames = 10
# outVidName = "output/res_Canvas.yuv"
# modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
# BATCHSIZE = 16

# baseCompPath = 'dataset/'
# vidName = "KristenAndSara_10frames_512x256.yuv420p"
# _comp = os.path.join(baseCompPath,vidName)
# _width = 512
# _height = 256
# _numFrames = 8
# _numFramesRestored = _numFrames
# outVidName = "output/res_Krissy.yuv"
# modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
# BATCHSIZE = 16

#alphas = [0.01, 0.001, 0.0001, 1e-05, 5e-06, 2.5e-06, 1.5e-06 ,1e-06]
#alphas = [1.0]
_alpha = 2.5e-06
#for _alpha in alphas:
print("alpha = ", _alpha)
seconds = time.time()
RGBvideo = readYUV2RGB(_comp,(_width,_height),scale=255.0)
#RGBvideo = RGBvideo[:2]
RGBvideo = np.insert(RGBvideo,0,RGBvideo[0,:,:,:],axis=0)
RGBvideo = np.insert(RGBvideo,-1,RGBvideo[-1,:,:,:],axis=0)

# get the number of patches for a frame
_testPatch = deconstruct(RGBvideo[0,:,:,:])
numPatches = _testPatch.shape[0]

print(numPatches)

# get patches from frames at time t, t-1 and t+1
patches_t = np.empty((numPatches *(RGBvideo.shape[0]-2),192,192,3))
patches_tmin1 = np.empty((numPatches *(RGBvideo.shape[0]-2),192,192,3))
patches_tplus1 = np.empty((numPatches *(RGBvideo.shape[0]-2),192,192,3))

# we build patches_t from slicing between the original start and end of array
for _frame in range(1,RGBvideo.shape[0]-1):
    patches_t[(_frame-1) * numPatches : ((_frame-1) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

# building patches_tmin1
for _frame in range(0,RGBvideo.shape[0]-2):
    patches_tmin1[(_frame) * numPatches : ((_frame) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

for _frame in range(2,RGBvideo.shape[0]):
    patches_tplus1[(_frame-2) * numPatches : ((_frame-2) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])


print(numPatches)
# old approach, kept running out of memory
# shape is now (N*numFrames) x 192 x192 x 9 where N is the number of patches obtained from deconstruct
#inputPatches = np.concatenate((patches_tmin1,patches_t,patches_tplus1), axis=-1)

# patches_tmin1 = tf.convert_to_tensor(patches_tmin1)
# patches_t = tf.convert_to_tensor(patches_t)
# patches_tplus1 = tf.convert_to_tensor(patches_tplus1)
# inputPatches = tf.concat([patches_tmin1,patches_t,patches_tplus1],-1)
# NB: Scale inputs from 0-255 to 0-1 for the new weights
#inputPatches = inputPatches/255.0

outputPatches = np.empty((numPatches * _numFrames,192,192,3))
hrNetModel = hrNet(2, [32, 64, 128, 256], 5).model()
# hrNetModel.load_weights(f'results/HRNET_CAMBI_ALPHA{_alpha}.h5')
hrNetModel.load_weights('results/HRNET_CAMBI_ALPHA2.5e-06_FINAL.h5')


print(f"GPUS: {tf.config.list_physical_devices('GPU')}")
print(_comp)

NUMSTEPS = (numPatches * _numFrames) // BATCHSIZE
print(NUMSTEPS)
for _step in range(NUMSTEPS):
    #print("Step" , _step)
    inputPatch = np.concatenate((patches_tmin1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_t[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_tplus1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:]), axis=-1)
    # inputPatch = inputPatches[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:]
    outputPatches[_step * BATCHSIZE : _step * BATCHSIZE + BATCHSIZE,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)
if (numPatches * _numFrames) > (BATCHSIZE * NUMSTEPS):
    # Patches left to process
    inputPatch = np.concatenate((patches_tmin1[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_t[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_tplus1[NUMSTEPS * BATCHSIZE: -1,:,:,:]), axis=-1)
    #inputPatch = inputPatches[NUMSTEPS * BATCHSIZE: -1,:,:,:]
    outputPatches[NUMSTEPS * BATCHSIZE: -1,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)

print("TEST")
# outputVideoRGB = np.empty((_numFrames,_height,_width,3))
outputVideoYUV = np.empty((_numFramesRestored,_height,_width,3))

for _frame in range(_numFramesRestored):
    #outputVideoRGB[_frame,:,:,:] = reconstruct(outputPatches[_frame * numPatches : _frame * numPatches + numPatches], RGBvideo[0],192)
    outputVideoPerFrame = reconstruct(outputPatches[_frame * numPatches : _frame * numPatches + numPatches], RGBvideo[0],192)
    outputVideoYUV[_frame,:,:,:] = RGB2YUV(np.expand_dims(outputVideoPerFrame,axis=0)) * 255.0
print("TEST 2")

#outputVideoYUV = RGB2YUV(outputVideoRGB)
print("TEST 3")

# outputVideoYUV = outputVideoYUV * 255.0
print("TEST 4")

Yout = outputVideoYUV[:,:,:,0]; Uout = outputVideoYUV[:,:,:,1]; Vout = outputVideoYUV[:,:,:,2]; 
#savePath = f'alphaValidation/alpha{(_alpha)}/' 
savePath = 'testSetOutput/' 
writeYUV420(savePath+"res_"+vidName,np.uint8(Yout),np.uint8(Uout),np.uint8(Vout),downsample=True)
seconds_2 = time.time()
print("Time taken =", seconds_2 - seconds)