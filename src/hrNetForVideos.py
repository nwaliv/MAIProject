from src.auxFunctions import readFrames, deconstruct, reconstruct
import numpy as np
import tensorflow as tf
from library import hrNet

class hrNetVideo:
    def __init__(self, video_name: str, vidResolution: tuple, vidNumFrames: int):
        self.video_name = video_name
        self.vidResolution = vidResolution
        self.vidNumFrames = vidNumFrames

    def restore_frame(self, t: int):

        frame_tmin1,frame_t,frame_tplus1 = readFrames(self.video_name,self.vidResolution,t,self.vidNumFrames)

        window_tmin1 = deconstruct(frame_tmin1,192)
        window_t = deconstruct(frame_t,192)
        window_tplus1 = deconstruct(frame_tplus1,192)

        artifactReductionModel = hrNet(2, [32, 64, 128, 256], 5).model()
        # Loads in the current weights, will be changed later after training on dataset
        artifactReductionModel.load_weights('modelWeights/287_HRNET.h5')

        inputPatches = np.concatenate((window_tmin1, window_t, window_tplus1), axis=-1)
        inputPatches = np.expand_dims(inputPatches, axis=0)
        numPatches = inputPatches.shape[1]
        inputPatches = tf.convert_to_tensor(inputPatches)

        outputPatches = np.empty((1,numPatches,192,192,3))
        for i in range(numPatches):
            outputPatches[:,i,:,:,:] = artifactReductionModel(inputPatches[:,i,:,:,:],training=False)

        outputFrame = reconstruct(outputPatches[0],frame_t,192)

        return outputFrame