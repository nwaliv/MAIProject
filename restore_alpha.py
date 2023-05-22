import pandas as pd
from src.hrNetRestoreFunctions import restore_clip

alphas = [1.0, 0.01, 0.001, 0.0001, 1e-05, 5e-06, 2.5e-06, 1.5e-06 ,1e-06]

for _alpha in alphas:
    print("alpha = ", _alpha)
    savePath = f'alphaValidation/alpha{(_alpha)}/'
    baseCompPath = 'validSetDegClip/'
    modelWeights = f'results/HRNET_CAMBI_ALPHA{_alpha}.h5'

    DATAFRAME = pd.read_csv('dataFrames/trainVideoSet.csv')
    vmaf_lim = 75
    cambi_lim = 10
    DATAFRAME = DATAFRAME[DATAFRAME['VMAF'] >= vmaf_lim]
    DATAFRAME = DATAFRAME[DATAFRAME['CAMBI'] >= cambi_lim]
    DATAFRAME = DATAFRAME.reset_index(drop=True)
    SAMPLE = DATAFRAME.sample(n=10, replace=False,random_state=10)
    SAMPLE = SAMPLE.reset_index(drop=True)

    for idx, _row in SAMPLE.iterrows():
        # get height/width of original video

        vidName = _row['Comp']
        _height = _row['Height']
        _width = _row['Width']
        _numFrames = _row['NumFrames']
        BATCHSIZE = 16
        print(vidName)

        restore_clip(baseCompPath,vidName,savePath,_width,_height,10,modelWeights)
