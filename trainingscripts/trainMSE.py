import tensorflow as tf 
from tensorflow import keras 
import tensorflow_addons as tfa 
import os 
from library import * 
from library.GeneralOps import readYUV420, YUV2RGB, RGB2YUV_TF, readYUV420RangePatches
from library.CriticModel import ProxyNetwork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import json 

patchSize=192
modelName = 'HRNET_MSE'


outputJson = f'results/{modelName}.json'
# the generator model will be hrNet
generator = hrNet(2, [32, 64, 128, 256], 5).model()

# Define the optimizers and loss function
optimizerGenerator = keras.optimizers.RMSprop(learning_rate=1e-4)
EPOCHS = 10
LossFunc = keras.losses.MeanSquaredError()
train_metric_gen_mse = keras.metrics.MeanSquaredError()

print(f"GPUS: {tf.config.list_physical_devices('GPU')}")

# Define the training functions
@tf.function
def train_step_gen(_x, _y):
    # _x is the degraded image, _y is the reference image
    with tf.GradientTape() as tape:
        # Generate the prediction
        _genPred = generator([_x], training=True)
        _genLoss = LossFunc(_genPred, _y)
    # Apply the gradients to the generator
    grads = tape.gradient(_genLoss, generator.trainable_weights)
    optimizerGenerator.apply_gradients(zip(grads, generator.trainable_weights))
    return _genPred

# DATAFRAME
BATCHSIZE = 4
# read in the training data
DATAFRAME = pd.read_csv('dataFrames/trainVideoSet_perframeCambi_maxFrame.csv')
DATAFRAME = DATAFRAME[DATAFRAME['FrameNum'] != 0]
DATAFRAME = DATAFRAME[DATAFRAME['FrameNum'] != DATAFRAME['MaxFrame']]
DATAFRAME = DATAFRAME.reset_index(drop=True)

baseRefPath = '/data/nwaliv/trainVideoSetRef/'
baseCompPath = '/data/nwaliv/trainVideoSetDeg/'

Loss = {'MSELoss' : []}

# the training loop
for _epoch in range(EPOCHS):
    # shuffle the dataframe
    indexes = np.arange(0, DATAFRAME.shape[0], 1)
    np.random.shuffle(indexes)
    _df = DATAFRAME.iloc[indexes].copy()
    # get the number of steps
    numSteps = _df.shape[0]//BATCHSIZE
    for step in range(numSteps): 
        # get the batch
        SAMPLE = _df.sample(n=BATCHSIZE, replace=False)
        _df = _df.drop(index=SAMPLE.index.tolist())
        SAMPLE = SAMPLE.reset_index(drop=True)
        # initialize the input and target arrays
        X_in = np.zeros((BATCHSIZE, patchSize, patchSize, 9))
        Y_target = np.zeros((BATCHSIZE, patchSize, patchSize, 3))
        # loop through the batch
        for idx, _row in SAMPLE.iterrows():
            # get height/width of original video
            _height = _row['Height']
            _width = _row['Width']
            _frameNum = _row['FrameNum']
            _comp = os.path.join(baseCompPath, _row['Comp'])
            _ref = os.path.join(baseRefPath, _row['Ref'])
            # generate random patch location 
            randHeight = np.random.randint(0,_height-patchSize)
            randWidth = np.random.randint(0,_width-patchSize)
            _xY, _xU, _xV = readYUV420RangePatches(_comp,(_width,_height),(_frameNum-1,_frameNum+1),(randWidth,randHeight),(patchSize,patchSize),True)
            _xYUV = np.stack([_xY, _xU, _xV], -1)
            _xRGB = YUV2RGB(_xYUV)/255.0

            _yY, _yU, _yV = readYUV420RangePatches(_ref,(_width,_height),(_frameNum,_frameNum),(randWidth,randHeight),(patchSize,patchSize),True)
            _yYUV = np.stack((_yY,_yU,_yV),axis=-1)
            _yRGB = YUV2RGB(_yYUV)/255.0

            if random.random() < 0.50:
                _channels = [0, 1, 2]
                random.shuffle(_channels)
                _xRGB = _xRGB[...,_channels]
                _yRGB = _yRGB[...,_channels]

            _xRGB = np.concatenate((_xRGB[0], _xRGB[1], _xRGB[2]), axis=-1).reshape(1,patchSize,patchSize,9)

            # add the data to the arrays
            X_in[idx] = _xRGB[0]
            Y_target[idx] = _yRGB[0]
        X_in, Y_target = X_in.astype(np.float32), Y_target.astype(np.float32)

        # train the generator
        _genPred = train_step_gen(X_in, Y_target)
        train_metric_gen_mse.update_state(_genPred, Y_target)

        if step % 1000 == 0:
            print(f'Epoch: {_epoch}, Step: {step}, MSE: {train_metric_gen_mse.result()}')
    Loss['MSELoss'].append(float(train_metric_gen_mse.result()))
    train_metric_gen_mse.reset_states()

generator.save_weights(f'results/{modelName}.h5')

with open(outputJson, "w") as outfile:
    json.dump(Loss, outfile)