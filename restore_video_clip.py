from library import hrNet
from library.GeneralOps import RGB2YUV, writeYUV420, readYUV2RGB
from src.auxFunctions import deconstruct, reconstruct
import numpy as np
import tensorflow as tf
import os
import time

baseCompPath = '/home/nwaliv/bandon/nwaliv/testVideoSetDeg/'
vidName = "wipe_3840x2160_5994_10bit_420_6309-6439_AV1_CRF38_.yuv"
_comp = os.path.join(baseCompPath,vidName)
_width = 1920
_height = 1080
_numFrames = 120
_numFramesRestored = 10
#modelWeights = 'results/HRNET_CAMBI_ALPHA2.5e-06_FINAL.h5'
modelWeights = 'results/HRNET_CAMBI_ALPHA1.0.h5'
#modelWeights = 'results/HRNET_MSE.h5'

BATCHSIZE = 4

seconds = time.time()
RGBvideo = readYUV2RGB(_comp,(_width,_height),scale=255.0)
RGBvideo = np.insert(RGBvideo,0,RGBvideo[0,:,:,:],axis=0)
RGBvideo = np.insert(RGBvideo,-1,RGBvideo[-1,:,:,:],axis=0)

# get the number of patches for a frame
_testPatch = deconstruct(RGBvideo[0,:,:,:])
numPatches = _testPatch.shape[0]

print(numPatches)

# get patches from frames at time t, t-1 and t+1
patches_t = np.empty((numPatches *(_numFramesRestored),192,192,3))
patches_tmin1 = np.empty((numPatches *(_numFramesRestored),192,192,3))
patches_tplus1 = np.empty((numPatches *(_numFramesRestored),192,192,3))

# we build patches_t from slicing between the original start and end of array
for _frame in range(1,_numFramesRestored+1):
    patches_t[(_frame-1) * numPatches : ((_frame-1) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

# building patches_tmin1
for _frame in range(0,_numFramesRestored):
    patches_tmin1[(_frame) * numPatches : ((_frame) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

for _frame in range(2,_numFramesRestored+2):
    patches_tplus1[(_frame-2) * numPatches : ((_frame-2) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])


print("TEST 1")

outputPatches = np.empty((numPatches * _numFramesRestored,192,192,3))
hrNetModel = hrNet(2, [32, 64, 128, 256], 5).model()
hrNetModel.load_weights(modelWeights)


print(f"GPUS: {tf.config.list_physical_devices('GPU')}")
print(_comp)

NUMSTEPS = (numPatches * _numFramesRestored) // BATCHSIZE
print(NUMSTEPS)
for _step in range(NUMSTEPS):
    inputPatch = np.concatenate((patches_tmin1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_t[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_tplus1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:]), axis=-1)
    outputPatches[_step * BATCHSIZE : _step * BATCHSIZE + BATCHSIZE,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)

# Patches left to process
if (numPatches * _numFramesRestored) > (BATCHSIZE * NUMSTEPS):
    inputPatch = np.concatenate((patches_tmin1[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_t[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_tplus1[NUMSTEPS * BATCHSIZE: -1,:,:,:]), axis=-1)
    outputPatches[NUMSTEPS * BATCHSIZE: -1,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)

print("TEST 2")

outputVideoYUV = np.empty((_numFramesRestored,_height,_width,3))

for _frame in range(_numFramesRestored):
    outputVideoPerFrame = reconstruct(outputPatches[_frame * numPatches : _frame * numPatches + numPatches], RGBvideo[0],192)
    outputVideoYUV[_frame,:,:,:] = RGB2YUV(np.expand_dims(outputVideoPerFrame,axis=0)) * 255.0
print("TEST 3")


Yout = outputVideoYUV[:,:,:,0]; Uout = outputVideoYUV[:,:,:,1]; Vout = outputVideoYUV[:,:,:,2]; 

savePath = 'testSetRestored/' 
writeYUV420(savePath+"res1.0_"+vidName,np.uint8(Yout),np.uint8(Uout),np.uint8(Vout),downsample=True)
seconds_2 = time.time()
print("Time taken =", seconds_2 - seconds)

def restore_video():
    return None