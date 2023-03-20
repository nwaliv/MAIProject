import tensorflow as tf 
from library.GeneralOps import writeYUV420, RGB2YUV, runTerminalCmd
import numpy as np
import os 
import pandas as pd 

def returnMetric(patchesDeg, patchesRef, processingPath="processing/", height=192, width=192):
    patchesRefNP_RGB = patchesRef
    if tf.is_tensor(patchesRefNP_RGB):
        patchesRefNP_RGB = patchesRefNP_RGB.numpy()
    patchesRefNP_RGB = np.clip(np.rint(patchesRefNP_RGB*255),0,255)
    patchesRefNP_YUV = np.clip(np.rint(RGB2YUV(patchesRefNP_RGB)),0,255).astype(np.uint8)
    videoFilePathRef = os.path.join(os.getcwd(), processingPath, "patchesRef.yuv")
    writeYUV420(videoFilePathRef, patchesRefNP_YUV[:,:,:,0], patchesRefNP_YUV[:,:,:,1], patchesRefNP_YUV[:,:,:,2])

    patchesDegNP_RGB = patchesDeg
    if tf.is_tensor(patchesDegNP_RGB):
        patchesDegNP_RGB = patchesDegNP_RGB.numpy()
    patchesDegNP_RGB = np.clip(np.rint(patchesDegNP_RGB*255),0,255)
    patchesDegNP_YUV = np.clip(np.rint(RGB2YUV(patchesDegNP_RGB)),0,255).astype(np.uint8)
    videoFilePathDeg = os.path.join(os.getcwd(), processingPath, "patchesDeg.yuv")
    writeYUV420(videoFilePathDeg, patchesDegNP_YUV[:,:,:,0], patchesDegNP_YUV[:,:,:,1], patchesDegNP_YUV[:,:,:,2])

    outputCSVPath = os.path.join(os.getcwd(), processingPath, "scores.csv")
    runTerminalCmd(f"vmaf -r {videoFilePathRef} -d {videoFilePathDeg} -w {width} -h {height} -p 420 -b 8 --csv -o {outputCSVPath} --feature cambi -q")
    actualScores = pd.read_csv(outputCSVPath)['cambi']
    actualScores = np.expand_dims(actualScores, -1).astype(np.float32)
    return actualScores