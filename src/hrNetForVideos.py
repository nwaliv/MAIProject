from src.auxFunctions import readFrames, deconstruct, reconstruct
from src.rgb2yuv_yuv2rgb import RGB2YUV
from library.GeneralOps import writeYUV420
import numpy as np
import tensorflow as tf
from library import hrNet

class hrNetVideo:
    def __init__(self, video_name: str, vidResolution: tuple, vidNumFrames: int, modelWeights: str):
        self.video_name = video_name
        self.vidResolution = vidResolution
        self.vidNumFrames = vidNumFrames
        self.modelWeights = modelWeights

    def restore_frame(self, t: int):

        frame_tmin1,frame_t,frame_tplus1 = readFrames(self.video_name,self.vidResolution,t,self.vidNumFrames)

        window_tmin1 = deconstruct(frame_tmin1,192)
        window_t = deconstruct(frame_t,192)
        window_tplus1 = deconstruct(frame_tplus1,192)

        artifactReductionModel = hrNet(2, [32, 64, 128, 256], 5).model()
        # Loads in the current weights
        artifactReductionModel.load_weights(self.modelWeights)

        inputPatches = np.concatenate((window_tmin1, window_t, window_tplus1), axis=-1)
        inputPatches = np.expand_dims(inputPatches, axis=0)
        numPatches = inputPatches.shape[1]
        inputPatches = tf.convert_to_tensor(inputPatches)

        outputPatches = np.empty((1,numPatches,192,192,3))
        for patch in range(numPatches):
            outputPatches[:,patch,:,:,:] = artifactReductionModel(inputPatches[:,patch,:,:,:],training=False)

        outputFrame = reconstruct(outputPatches[0],frame_t,192)

        return outputFrame

    def restore_video(self):

        artifactReductionModel = hrNet(2, [32, 64, 128, 256], 5).model()
        # Loads in the current weights
        artifactReductionModel.load_weights(self.modelWeights)

        restoredY = np.empty((self.vidNumFrames,self.vidResolution[1],self.vidResolution[0]))
        restoredU = np.empty((self.vidNumFrames,self.vidResolution[1],self.vidResolution[0]))
        restoredV = np.empty((self.vidNumFrames,self.vidResolution[1],self.vidResolution[0]))
        restoredFrames = np.empty((self.vidNumFrames,self.vidResolution[1],self.vidResolution[0],3))

        #self.vidNumFrames
        for frame in range(2):
            print("Frame ", frame)
            frame_tmin1,frame_t,frame_tplus1 = readFrames(self.video_name,self.vidResolution,frame,self.vidNumFrames)

            window_tmin1 = deconstruct(frame_tmin1,192)
            window_t = deconstruct(frame_t,192)
            window_tplus1 = deconstruct(frame_tplus1,192)

            inputPatches = np.concatenate((window_tmin1, window_t, window_tplus1), axis=-1)
            inputPatches = np.expand_dims(inputPatches, axis=0)
            numPatches = inputPatches.shape[1]
            inputPatches = tf.convert_to_tensor(inputPatches)

            outputPatches = np.empty((1,numPatches,192,192,3))
            for patch in range(numPatches):
                outputPatches[:,patch,:,:,:] = artifactReductionModel(inputPatches[:,patch,:,:,:],training=False)

            outputFrame = reconstruct(outputPatches[0],frame_t,192)
            restoredFrames[frame,:,:,:] = outputFrame[:,:,:]

            outputFrameYUV = RGB2YUV(outputFrame)

            Yout = outputFrameYUV[:,:,0]
            Uout = outputFrameYUV[:,:,1]
            Vout = outputFrameYUV[:,:,2]

            restoredY[frame,:,:] = Yout
            restoredU[frame,:,:] = Uout
            restoredV[frame,:,:] = Vout

        writeYUV420("output/"+ self.video_name[30:],np.uint8(restoredY),np.uint8(restoredU),np.uint8(restoredV),downsample=True)

        return restoredFrames