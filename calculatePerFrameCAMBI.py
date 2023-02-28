# Code to calculate the cambi scores of the training dataset per frame
# to analyse which frames are worth training on
# requires ffmpeg to be working aswell
import subprocess
import json
import pandas as pd
import time 

def runCommand(sumCommand: str):
    subprocess.run(sumCommand, shell=True)

pathRef = '/data/nwaliv/trainVideoSetRef/'
pathComp = '/data/nwaliv/trainVideoSetDeg/'
refName = []
compName = []
frameNumber = []
cambiScore = []
vmaf_lim = 75
cambi_lim = 10
df = pd.read_csv('dataFrames/trainVideoSet.csv')
df = df[df['VMAF'] >= vmaf_lim]
df = df[df['CAMBI'] >= cambi_lim]
df = df.reset_index(drop=True)
df_perframe = {
    'Ref': [],
    'Comp': [],
    'Height': [],
    'Width': [],
    'FrameNum': [],
    'Cambi': []
}

for cnt, index in enumerate(range(df.shape[0])):
    _start = time.time()
    refFile = pathRef + df["Ref"][index]
    compFile = pathComp + df["Comp"][index]
    command = f"""vmaf -r {refFile} -d {compFile} -h {df["Height"][index]} -w {df["Width"][index]} -p 420 -b 8 --no_prediction --feature cambi --csv -o output.csv"""
    #print(command)  
    runCommand(command)
    _df = pd.read_csv('output.csv')
    _frames = _df['Frame'].tolist()
    _cambi = _df['cambi'].tolist()
    _ref = [df["Ref"][index]]*len(_frames)
    _comp = [df["Comp"][index]]*len(_frames)
    _height = [df["Height"][index]]*len(_frames)
    _width = [df["Width"][index]]*len(_frames)

    df_perframe['Ref'].extend(_ref)
    df_perframe['Comp'].extend(_comp)
    df_perframe['Height'].extend(_height)
    df_perframe['Width'].extend(_width)
    df_perframe['FrameNum'].extend(_frames)
    df_perframe['Cambi'].extend(_cambi)

    print(f'Finished: {refFile}, \t \t Percentage {round(cnt*100/df.shape[0],5)} \t \t Time:{round(time.time()-_start,3)}')

df_perframe = pd.DataFrame(df_perframe)
df_perframe.to_csv('dataFrames/trainVideoSet_perframeCambi.csv')