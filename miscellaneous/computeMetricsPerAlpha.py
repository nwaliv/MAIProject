import pandas as pd
import os
from library.GeneralOps import runTerminalCmd



alphas = [1.0, 0.01, 0.001, 0.0001, 1e-05, 5e-06, 2.5e-06, 1.5e-06 ,1e-06]

for _alpha in alphas:
    baseRefPath = 'validSetRefClip'
    baseCompPath = f'alphaValidation/alpha{_alpha}/'

    DATAFRAME = pd.read_csv('dataFrames/trainVideoSet.csv')
    vmaf_lim = 75
    cambi_lim = 10
    DATAFRAME = DATAFRAME[DATAFRAME['VMAF'] >= vmaf_lim]
    DATAFRAME = DATAFRAME[DATAFRAME['CAMBI'] >= cambi_lim]
    DATAFRAME = DATAFRAME.reset_index(drop=True)
    SAMPLE = DATAFRAME.sample(n=10, replace=False,random_state=10)
    SAMPLE = SAMPLE.reset_index(drop=True)

    df_peralpha = {
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

    for idx, _row in SAMPLE.iterrows():
        # get height/width of original video
        _height = _row['Height']
        _width = _row['Width']
        _comp = os.path.join(baseCompPath,'res_'+_row['Comp'])
        _ref = os.path.join(baseRefPath, _row['Ref'])
        #print(_comp)
        runTerminalCmd(f"vmaf -r {_ref} -d {_comp} -w {_width} -h {_height} -p 420 -b 8 --csv -o processing/evals.csv --feature psnr --feature float_ssim --feature cambi -q")
        _cambi = pd.read_csv('processing/evals.csv')['cambi'].mean()
        _psnr_y = pd.read_csv('processing/evals.csv')['psnr_y'].mean()
        _psnr_cb = pd.read_csv('processing/evals.csv')['psnr_cb'].mean()
        _psnr_cr = pd.read_csv('processing/evals.csv')['psnr_cr'].mean()
        _ssim = pd.read_csv('processing/evals.csv')['float_ssim'].mean()
        _vmaf = pd.read_csv('processing/evals.csv')['vmaf'].mean()
        df_peralpha['Ref'].append(_row['Ref'])
        df_peralpha['Comp'].append('res_'+_row['Comp'])
        df_peralpha['Height'].append(_height)
        df_peralpha['Width'].append(_width)
        df_peralpha['PSNR_Y'].append(_psnr_y)
        df_peralpha['PSNR_Cb'].append(_psnr_cb)
        df_peralpha['PSNR_Cr'].append(_psnr_cr)
        df_peralpha['SSIM'].append(_ssim)
        df_peralpha['VMAF'].append(_vmaf)
        df_peralpha['CAMBI'].append(_cambi)
    df_peralpha = pd.DataFrame(df_peralpha)
    df_peralpha.to_csv(f'alphaValidation/alpha{_alpha}/alphaScores.csv')
