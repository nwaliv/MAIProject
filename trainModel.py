# Code to train the model

# Importing the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from src import hrNet
from src import readFrames, deconstruct, createPatches

# To create an array of rolling patches with shape (BatchSize,192,192,9)
# 1 is the batch size, 192 is the patch size, 9 is the number of channels
inputPatchesRollingTotal = np.ones((1,192,192,9))

# Read the csv file
df = pd.read_csv("trainVideoSet.csv")
# Iterate through the csv file
for videoNum in range(df.shape[0]):
    # Read the video
    inputVideo = df["Comp"][videoNum]
    vidResolution = (df["Height"][videoNum],df["Width"][videoNum])
    vidNumFrames = df["NumFrames"][videoNum]

    # Iterate through the frames
    for frame in range(vidNumFrames):
        # Read the frames
        frame_tmin1, frame_t, frame_tplus1 = readFrames(inputVideo, vidResolution,frame,vidNumFrames)
        # Create the patches
        # The patches are of shape (BatchSize,192,192,3)
        window_tmin1 = createPatches(frame_tmin1,192); window_t = createPatches(frame_t,192); window_tplus1 = createPatches(frame_tplus1,192)
        # Concatenate the patches
        # The patches are of shape (BatchSize,192,192,9)
        inputPatches = np.concatenate((window_tmin1, window_t, window_tplus1), axis=-1)
        # Append the patches to the rolling total
        inputPatchesRollingTotal = np.vstack((inputPatchesRollingTotal,inputPatches))

# Delete the first row of ones
inputPatchesRollingTotal = np.delete(inputPatchesRollingTotal,(0),axis=0) 

# Train the model   
hrNetModel = hrNet(2, [32, 64, 128, 256], 5).model()
# Compile the model
hrNetModel.compile(run_eagerly=True, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.mse)
hrNetModel.fit(inputPatchesRollingTotal, inputPatchesRollingTotal, epochs=10, batch_size=1, shuffle=True)
# Save the model
hrNetModel.save("hrNetModel.h5")
