# Code to calculate the cambi scores of the training dataset per frame
# to analyse which frames are worth training on
# requires ffmpeg to be working aswell
import subprocess
import json
import pandas as pd

def runCommand(sumCommand: str):
    subprocess.run(sumCommand)

pathRef = '/home/nwaliv/bandon/nwaliv/rawvideos'
pathComp = '/home/nwaliv/bandon/nwaliv/trainVideoSet'
refName = []
compName = []
frameNumber = []
cambiScore = []
df = pd.read_csv("trainVideoSet.csv")

for index in range(df.shape[0]):
    refFile = file + df["Ref"][index]
    compFile = file + df["Comp"][index]
    resolution = str(df["Width"][index])+"x"+str(df["Height"][index])
    command = f"""ffmpeg -s {resolution} -i {refFile} -s {resolution} -i {compFile} -lavfi libvmaf="feature='name=psnr|name=float_ssim|name=cambi':log_fmt=json:log_path=output.json" -f null -"""
    runCommand(command)
    f = open('output.json')
    file = json.load(f)
    
    for frame in range(df["NumFrames"][index]):
        refName.append(df["Ref"][index])
        compName.append(df["Comp"][index])
        frameNumber.append(frame)
        cambiScore.append(file["frames"][frame]['metrics']["cambi"])

data = {'Ref':refName, 'Comp':compName, 'FrameNum':frameNumber, 'cambi':cambiScore}
# Create a dataframe
dfPerFrame = pd.DataFrame(data)
dfPerFrame.head()
dfPerFrame.to_csv("trainVideoSetPerFrame.csv")  