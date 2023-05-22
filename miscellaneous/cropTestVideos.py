import pandas as pd
from library.GeneralOps import  readYUV420Range,writeYUV420
import os

# For copying the test ones
# DATAFRAME = pd.read_csv("dataFrames/testVideoSet.csv")

# compPath = '/home/nwaliv/bandon/ramsookd/data/rawVideos/'
# cropPath = 'testSetRefClip/'

# for cnt, index in enumerate(range(DATAFRAME.shape[0])):
#     print(DATAFRAME['Ref'][index])
#     _comp = DATAFRAME['Ref'][index]
#     _height = DATAFRAME['Height'][index]
#     _width = DATAFRAME['Width'][index]
#     _comp = os.path.join(compPath,DATAFRAME['Ref'][index])
#     Y,U,V = readYUV420Range(_comp,(_width,_height),(0,9),True)
#     _output = os.path.join(cropPath,DATAFRAME['Ref'][index])
#     writeYUV420(_output,Y,U,V,True)


Y,U,V = readYUV420Range("adaband_wipe_3840x2160_5994_10bit_420_6309-6439_AV1_CRF38_.yuv",(1920,1080),(0,9),True)
writeYUV420("ada_wipe_3840x2160_5994_10bit_420_6309-6439_AV1_CRF38_.yuv",Y,U,V,True)


# DATAFRAME = pd.read_csv('dataFrames/trainVideoSet.csv')
# vmaf_lim = 75
# cambi_lim = 10
# DATAFRAME = DATAFRAME[DATAFRAME['VMAF'] >= vmaf_lim]
# DATAFRAME = DATAFRAME[DATAFRAME['CAMBI'] >= cambi_lim]
# DATAFRAME = DATAFRAME.reset_index(drop=True)
# SAMPLE = DATAFRAME.sample(n=10, replace=False,random_state=10)
# SAMPLE = SAMPLE.reset_index(drop=True)

# compPath = '/home/nwaliv/bandon/ramsookd/data/rawVideos/'
# cropPath = 'validSetRefClip/'

# for cnt, index in enumerate(range(SAMPLE.shape[0])):
#     print(SAMPLE['Ref'][index])
#     _comp = SAMPLE['Ref'][index]
#     _height = SAMPLE['Height'][index]
#     _width = SAMPLE['Width'][index]
#     _comp = os.path.join(compPath,SAMPLE['Ref'][index])
#     Y,U,V = readYUV420Range(_comp,(_width,_height),(0,9),True)
#     _output = os.path.join(cropPath,SAMPLE['Ref'][index])
#     writeYUV420(_output,Y,U,V,True)