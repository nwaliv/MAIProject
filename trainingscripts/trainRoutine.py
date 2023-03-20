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

# Test reading in a reference and degraded video
refFile = 'Videos/REF.yuv'
degFile = 'Videos/DEG.yuv'

# Read in the first 64 frames of the video
numFrames = 64

# Read in the YUV files
refY, refU, refV = readYUV420(refFile, (512,512), True)
refYUV = np.stack([refY, refU, refV], -1)[:numFrames]
refRGB = YUV2RGB(refYUV)/255

degY, degU, degV = readYUV420(degFile, (512,512), True)
degYUV = np.stack([degY, degU, degV], -1)[:numFrames]
degRGB = YUV2RGB(degYUV)/255

# Read in the VMAF model
critic = ProxyNetwork().model()
# the generator model will be hrNet
generator = hrNet().model()

# Define the optimizers and loss function
optimizerCritic = keras.optimizers.Adam(learning_rate=1e-3)
optimizerGenerator = keras.optimizers.Adam(learning_rate=1e-3)
EPOCHS = 100
LossFunc = keras.losses.MeanSquaredError()

train_metric_critic = keras.metrics.MeanSquaredError()
train_metric_gen_mse = keras.metrics.MeanSquaredError()
train_metric_gen_vmaf = keras.metrics.MeanSquaredError()

# Define the training functions
@tf.function
def train_step_gen(_x, _y, _yLuma):
    # _x is the degraded image, _y is the reference image, _yLuma is the luma of the reference image
    with tf.GradientTape() as tape:
        # Generate the prediction
        _genPred = generator([_x], training=True)
        # Get the luma of the prediction
        _genPredLuma = tf.expand_dims(RGB2YUV_TF(_genPred)[:,:,:,0],-1)
        # Get the critic score for the prediction
        _criticPred = critic([_yLuma, _genPredLuma], training=False)
        # Define the upper bound for the critic score (will change in the future)
        _upper = tf.ones_like(_criticPred) * 100
        # Define the loss function for the generator as the sum of the critic score and the MSE
        _genLoss = LossFunc(_criticPred, _upper) + LossFunc(_genPred, _y)
    # Apply the gradients to the generator
    grads = tape.gradient(_genLoss, generator.trainable_weights)
    optimizerGenerator.apply_gradients(zip(grads, generator.trainable_weights))
    return _genPred, _criticPred, _upper, _genPredLuma

@tf.function
def train_step_critic(_yLuma, _score, _genPredLuma):
    # _yLuma is the luma of the reference image, _score is the critic score for the prediction, _genPredLuma is the luma of the prediction
    with tf.GradientTape() as tape:
        # Get the critic score for the prediction
        _pred = critic([_yLuma, _genPredLuma], training=True)
        # Define the loss function for the critic as the MSE between the critic score and the prediction
        _loss = LossFunc(_pred, _score)
    # Apply the gradients to the critic
    grads = tape.gradient(_loss, critic.trainable_weights)
    optimizerCritic.apply_gradients(zip(grads, critic.trainable_weights))
    return _pred

# DATAFRAME
# BATCHSIZE = 4
BATCHSIZE = 4
# read in the training data
DATAFRAME = pd.read_csv('dataFrames/trainVideoSet_perFrameCambi.csv')
# the training loop
for _epoch in range(EPOCHS):
    # shuffle the dataframe
    indexes = np.arange(0, DATAFRAME.shape[0], 1)
    np.random.shuffle(indexes)
    _df = DATAFRAME.iloc[indexes]
    # get the number of steps
    numSteps = _df.shape[0]//BATCHSIZE
    for step in range(numSteps):
        # get the batch
        SAMPLE = _df.sample(n=BATCHSIZE, replace=False)
        # initialize the input and target arrays
        X_in = np.zeros((BATCHSIZE, 192, 192, 9))
        Y_target = np.zeros((BATCHSIZE, 192, 192, 3))
        Y_targetLuma = np.zeros((BATCHSIZE, 192, 192, 1))
        # loop through the batch
        for idx, _row in enumerate(SAMPLE):
            # get height/width of original video
            _height = _row['Height']
            _width = _row['Width']
            _frameNum = _row['FrameNum']
            # generate random patch location 
            randHeight = np.random.randint(0,_x.shape[1]-192)
            randWidth = np.random.randint(0,_x.shape[2]-192)
            _x = readYUV420RangePatches(_row['Comp'],(_width,_height),(_frameNum-1,_frameNum+1),(randWidth,randHeight),(192,192),True)
            _y = readYUV420RangePatches(_row['Ref'],(_width,_height),(_frameNum-1,_frameNum+1),(randWidth,randHeight),(192,192),True)
            _x = np.stack((_x[0],_x[1],_x[2]),axis=-1)
            _x = YUV2RGB(_x)/255
            _x = np.concatenate((_frameNum -1, _frameNum, _frameNum +1), axis=-1).reshape(1,192,192,9)
            _y = _y[_frameNum].reshape(1,192,192,3)
            _y = YUV2RGB(_y)/255
            # Use readYUV420RangPAtches
            # Convert to RGB


            # read in the YUV files - this is the original code
            _x = readYUV420(_row['Comp'],(_row['Width'],_row['Height']),True)
            _y = readYUV420(_row['Ref'],(_row['Width'],_row['Height']),True)
            # prepare the data
            _x = np.stack((_x[0],_x[1],_x[2]),axis=-1)
            _x = YUV2RGB(_x)/255
            _x = np.concatenate((_x[_row['FrameNum']] -1, _x[_row['FrameNum']], _x[_row['FrameNum']] +1), axis=-1).reshape(1,_row['Width'],_row['Height'],9)
            _y = np.stack((_y[0],_y[1],_y[2]),axis=-1)
            _y = _y[_row['FrameNum']].reshape(1,_row['Width'],_row['Height'],3)
            _y = YUV2RGB(_y)/255
            # select a random 192x192 frames from the RGB video
            randHeight = np.random.randint(0,_x.shape[1]-192)
            randWidth = np.random.randint(0,_x.shape[2]-192)
            _x = _x[0, randHeight:randHeight+192, randWidth:randWidth+192, :]
            _y = _y[0, randHeight:randHeight+192, randWidth:randWidth+192, :]

            # add the data to the arrays
            X_in[idx] = _x
            Y_target[idx] = _y
            Y_targetLuma[idx] = RGB2YUV_TF(_y)[:,:,:,0].reshape(1,192,192,1)
        # train the generator
        _genPred, _criticPred, _upper, _genPredLuma = train_step_gen(X_in, Y_target, Y_targetLuma)
        _score = returnMetric(_genPred, _y, height=192, width=192)    
        _pred = train_step_critic(_yLuma, _score, _genPredLuma) 

        train_metric_critic.update_state(_pred, _score)
        train_metric_gen_mse.update_state(_genPred, _y)
        train_metric_gen_vmaf.update_state(_criticPred, _upper)

    # _DF = SHUFFLE(DATAFRAME)
    # NUMSTEPS = _DF.shape[0]//BATCHSIZE
    # for _step in range(NUMSTEPS):
    #   _SAMPLE = _DF.sample(n=BATCHSIZE, replacement=FALSE)

    #   X_IN = np.zeros((BATCHSIZE, 192, 192, 3))
    #   OTHER INPUTS
    #   Y_TARGET = np.zeros(())


    #   For idx, _row in enumerate(_SAMPLE):
    #       READ IN YUV FILES (_x)
    #       PREP DATA ETC 
    #       X_IN[idx] = _x
    #   TRAIN FUNCTIONS HERE 




    indexes = np.arange(0, refRGB.shape[0], 1)
    np.random.shuffle(indexes)
    refRGBShuffled = refRGB[indexes]
    degRGBShuffled = degRGB[indexes]
    for step in range(len(indexes)):
        _y = refRGBShuffled[[step]].astype(np.float32)
        _x = degRGBShuffled[[step]].astype(np.float32)
        _yLuma = np.expand_dims(refY[[step]],-1)/255



        _genPred, _criticPred, _upper, _genPredLuma = train_step_gen(_x, _y, _yLuma)
        _score = returnMetric(_genPred, _y, height=512, width=512)    
        _pred = train_step_critic(_yLuma, _score, _genPredLuma) 



        
        train_metric_critic.update_state(_pred, _score)
        train_metric_gen_mse.update_state(_genPred, _y)
        train_metric_gen_vmaf.update_state(_criticPred, _upper)

    print(f"Epoch: {_epoch}, gen MSE: {train_metric_gen_mse.result()}, gen VMAF: {train_metric_gen_vmaf.result()} ,Critic MSE: {train_metric_critic.result()}")
    train_metric_critic.reset_state()
    train_metric_gen_vmaf.reset_state()
    train_metric_gen_mse.reset_state()