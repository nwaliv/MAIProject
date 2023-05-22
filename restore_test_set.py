from src.hrNetRestoreFunctions import restore_clip
from library.GeneralOps import runTerminalCmd
import pandas as pd
import os

#Code for restoring the test set of videos
DATAFRAME = pd.read_csv("dataFrames/testVideoSet.csv")

compPath = 'testSetDegClip/'
outputPath ='testSetRestored/'

for cnt, index in enumerate(range(DATAFRAME.shape[0])):

    _comp = DATAFRAME['Comp'][index]
    print(_comp)
    _height = DATAFRAME['Height'][index]
    _width = DATAFRAME['Width'][index]
    _numFrames = 10
    modelWeights = 'results/HRNET_CAMBI_ALPHA2.5e-06_FINAL.h5'
    restore_clip(compPath,_comp,outputPath,_width,_height,_numFrames,modelWeights)

df_test_restored = {
    'Ref': [],
    'Comp': [],
    'Height': [],
    'Width': [],
    'PSNR_Y': [],
    'PSNR_Cb': [],
    'PSNR_Cr': [],
    'PSNR_Y': [],
    'SSIM': [],
    'VMAF': [],
    'CAMBI': []
    }

baseCompPath = 'testSetRestored/'
baseRefPath = 'testSetRefClip/'
for idx, _row in DATAFRAME.iterrows():
    # get height/width of original video
    _height = _row['Height']
    _width = _row['Width']
    _comp = os.path.join(baseCompPath,'res_'+_row['Comp'])
    _ref = os.path.join(baseRefPath, _row['Ref'])
    print(_comp)
    runTerminalCmd(f"vmaf -r {_ref} -d {_comp} -w {_width} -h {_height} -p 420 -b 8 --csv -o processing/evals.csv --feature psnr --feature float_ssim --feature cambi -q")
    _cambi = pd.read_csv('processing/evals.csv')['cambi'].mean()
    _psnr_y = pd.read_csv('processing/evals.csv')['psnr_y'].mean()
    _psnr_cb = pd.read_csv('processing/evals.csv')['psnr_cb'].mean()
    _psnr_cr = pd.read_csv('processing/evals.csv')['psnr_cr'].mean()
    _ssim = pd.read_csv('processing/evals.csv')['float_ssim'].mean()
    _vmaf = pd.read_csv('processing/evals.csv')['vmaf'].mean()
    df_test_restored['Ref'].append(_row['Ref'])
    df_test_restored['Comp'].append('res_'+_row['Comp'])
    df_test_restored['Height'].append(_height)
    df_test_restored['Width'].append(_width)
    df_test_restored['PSNR_Y'].append(_psnr_y)
    df_test_restored['PSNR_Cb'].append(_psnr_cb)
    df_test_restored['PSNR_Cr'].append(_psnr_cr)
    df_test_restored['SSIM'].append(_ssim)
    df_test_restored['VMAF'].append(_vmaf)
    df_test_restored['CAMBI'].append(_cambi)

df_test_restored = pd.DataFrame(df_test_restored)
df_test_restored.to_csv('dataFrames/hrNetScores.csv')