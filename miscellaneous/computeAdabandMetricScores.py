import pandas as pd
import os
from library.GeneralOps import runTerminalCmd
baseRefPath = '/home/nwaliv/bandon/ramsookd/data/rawVideos/'
baseCompPath = '/home/nwaliv/bandon/nwaliv/adabandVideos/'


df_adaband = {
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

DATAFRAME = pd.read_csv('dataFrames/testVideoSet.csv')
for _idx, _row in DATAFRAME.iterrows():
    _height = _row['Height']
    _width = _row['Width']
    _comp = os.path.join(baseCompPath,'adaband_'+_row['Comp'])
    _ref = os.path.join(baseRefPath, _row['Ref'])
    print('adaband_'+_row['Comp'])
    runTerminalCmd(f"vmaf -r {_ref} -d {_comp} -w {_width} -h {_height} -p 420 -b 8 --csv -o processing/evals.csv --feature psnr --feature float_ssim --feature cambi -q")
    _cambi = pd.read_csv('processing/evals.csv')['cambi'].mean()
    print(_cambi)
    _psnr_y = pd.read_csv('processing/evals.csv')['psnr_y'].mean()
    _psnr_cb = pd.read_csv('processing/evals.csv')['psnr_cb'].mean()
    _psnr_cr = pd.read_csv('processing/evals.csv')['psnr_cr'].mean()
    _ssim = pd.read_csv('processing/evals.csv')['float_ssim'].mean()
    _vmaf = pd.read_csv('processing/evals.csv')['vmaf'].mean()
    df_adaband['Ref'].append(_row['Ref'])
    df_adaband['Comp'].append('adaband_'+_row['Comp'])
    df_adaband['Height'].append(_height)
    df_adaband['Width'].append(_width)
    df_adaband['PSNR_Y'].append(_psnr_y)
    df_adaband['PSNR_Cb'].append(_psnr_cb)
    df_adaband['PSNR_Cr'].append(_psnr_cr)
    df_adaband['SSIM'].append(_ssim)
    df_adaband['VMAF'].append(_vmaf)
    df_adaband['CAMBI'].append(_cambi)
    
df_adaband = pd.DataFrame(df_adaband)
df_adaband.to_csv('dataFrames/adabandScores.csv')