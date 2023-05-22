from src.hrNetRestoreFunctions import restore_clip
import pandas as pd


# Sample video to run
# baseCompPath = 'dataset/'
# vidName = "canvas_480x272.yuv"
# _width = 480
# _height = 272
# _numFrames = 10
# outputPath = "output/"
# modelWeights = "results/HRNET_CAMBI_ALPHA1.0.h5"
# BATCHSIZE = 16

baseCompPath = 'testSetDegClip/'
vidName = "wipe_3840x2160_5994_10bit_420_6309-6439_AV1_CRF63_.yuv"
_width = 1920
_height = 1080
_numFramesRestored = 10
#outputPath = 'testSetRestored/'
outputPath = 'output/'
modelWeights = "results/HRNET_CAMBI_ALPHA2.5e-06_FINAL.h5"
BATCHSIZE = 4

restore_clip(baseCompPath,vidName,outputPath,_width,_height,_numFramesRestored,modelWeights)


# Code for restoring the test set of videos
# DATAFRAME = pd.read_csv("dataFrames/testVideoSet.csv")

# compPath = 'testSetDegClip/'
# outputPath ='testSetRestored/'

# for cnt, index in enumerate(range(DATAFRAME.shape[0])):

#     _comp = DATAFRAME['Comp'][index]
#     print(_comp)
#     _height = DATAFRAME['Height'][index]
#     _width = DATAFRAME['Width'][index]
#     _numFrames = 10
#     modelWeights = 'results/HRNET_CAMBI_ALPHA2.5e-06_FINAL.h5'
#     restore_clip(compPath,_comp,outputPath,_width,_height,_numFrames,modelWeights)