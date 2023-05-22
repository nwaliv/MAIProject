from library import hrNet
from library.GeneralOps import RGB2YUV, writeYUV420, readYUV2RGB
from src.auxFunctions import deconstruct, reconstruct
import numpy as np
import tensorflow as tf
import os
import time


def restore_clip(compPath: str, compName: str, outputPath: str, vidWidth: int, vidHeight: int, numFramesRestored: int, modelWeights: str, BATCHSIZE=4):

    seconds = time.time()

    _comp = os.path.join(compPath,compName)
    RGBvideo = readYUV2RGB(_comp,(vidWidth,vidHeight),scale=255.0)
    RGBvideo = np.insert(RGBvideo,0,RGBvideo[0,:,:,:],axis=0)
    RGBvideo = np.insert(RGBvideo,-1,RGBvideo[-1,:,:,:],axis=0)

    # get the number of patches for a frame
    _testPatch = deconstruct(RGBvideo[0,:,:,:])
    numPatches = _testPatch.shape[0]
    print("Number of patches per frame",numPatches)

    # get patches from frames at time t, t-1 and t+1
    patches_t = np.empty((numPatches *(numFramesRestored),192,192,3))
    patches_tmin1 = np.empty((numPatches *(numFramesRestored),192,192,3))
    patches_tplus1 = np.empty((numPatches *(numFramesRestored),192,192,3))

    # we build patches_t from slicing between the original start and end of array
    for _frame in range(1,numFramesRestored+1):
        patches_t[(_frame-1) * numPatches : ((_frame-1) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

    # building patches_tmin1
    for _frame in range(0,numFramesRestored):
        patches_tmin1[(_frame) * numPatches : ((_frame) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

    # building patches_tplus1
    for _frame in range(2,numFramesRestored+2):
        patches_tplus1[(_frame-2) * numPatches : ((_frame-2) * numPatches + numPatches),:,:,:] = deconstruct(RGBvideo[_frame,:,:,:])

    print("Retrieving Video Patches...")

    outputPatches = np.empty((numPatches * numFramesRestored,192,192,3))
    hrNetModel = hrNet(2, [32, 64, 128, 256], 5).model()
    hrNetModel.load_weights(modelWeights)

    print(f"GPUS: {tf.config.list_physical_devices('GPU')}")
    print("Video Name:",compName)

    NUMSTEPS = (numPatches * numFramesRestored) // BATCHSIZE
    print("Number of steps per Batchsize:",NUMSTEPS)

    for _step in range(NUMSTEPS):
        inputPatch = np.concatenate((patches_tmin1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_t[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:],patches_tplus1[_step * BATCHSIZE: _step * BATCHSIZE + BATCHSIZE,:,:,:]), axis=-1)
        outputPatches[_step * BATCHSIZE : _step * BATCHSIZE + BATCHSIZE,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)

    # Patches left to process
    if (numPatches * numFramesRestored) > (BATCHSIZE * NUMSTEPS):
        inputPatch = np.concatenate((patches_tmin1[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_t[NUMSTEPS * BATCHSIZE: -1,:,:,:],patches_tplus1[NUMSTEPS * BATCHSIZE: -1,:,:,:]), axis=-1)
        outputPatches[NUMSTEPS * BATCHSIZE: -1,:,:,:] = hrNetModel(tf.convert_to_tensor(inputPatch), training=False)

    print("Patches Restoration Completed....")

    outputVideoYUV = np.empty((numFramesRestored,vidHeight,vidWidth,3))

    for _frame in range(numFramesRestored):
        outputVideoPerFrame = reconstruct(outputPatches[_frame * numPatches : _frame * numPatches + numPatches], RGBvideo[0],192)
        outputVideoYUV[_frame,:,:,:] = RGB2YUV(np.expand_dims(outputVideoPerFrame,axis=0)) * 255.0

    print("Completed Building Frames from Patches ...")


    Yout = outputVideoYUV[:,:,:,0]; Uout = outputVideoYUV[:,:,:,1]; Vout = outputVideoYUV[:,:,:,2]; 

    writeYUV420(outputPath+"res_"+compName,np.uint8(Yout),np.uint8(Uout),np.uint8(Vout),downsample=True)
    seconds_2 = time.time()
    print("Time taken =", seconds_2 - seconds)
    return None