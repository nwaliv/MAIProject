import subprocess
import json
import pandas as pd
import time 

def runCommand(sumCommand: str):
    subprocess.run(sumCommand, shell=True)

vmaf_lim = 75
cambi_lim = 10
df = pd.read_csv('dataFrames/trainVideoSet.csv')
df = df[df['VMAF'] >= vmaf_lim]
df = df[df['CAMBI'] >= cambi_lim]
compFiles = df['Comp'].to_list()

for _file in compFiles:
    runCommand(f'rsync --progress /home/nwaliv/bandon/nwaliv/trainVideoSet/{_file} /data/nwaliv/trainVideoSetDeg/')

refFiles = df['Ref'].unique().tolist()

for _file in refFiles:
    runCommand(f'rsync --progress /home/nwaliv/bandon/ramsookd/data/rawVideos/{_file} /data/nwaliv/trainVideoSetRef/')