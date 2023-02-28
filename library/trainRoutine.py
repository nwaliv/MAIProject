import tensorflow as tf 
from tensorflow import keras 
import tensorflow_addons as tfa 
import os 
from library import * 
from library.GeneralOps import readYUV420, YUV2RGB, RGB2YUV_TF
from library.CriticModel import ProxyNetwork
import matplotlib.pyplot as plt
import numpy as np

refFile = 'Videos/REF.yuv'
degFile = 'Videos/DEG.yuv'

numFrames = 64

refY, refU, refV = readYUV420(refFile, (512,512), True)
refYUV = np.stack([refY, refU, refV], -1)[:numFrames]
refRGB = YUV2RGB(refYUV)/255

degY, degU, degV = readYUV420(degFile, (512,512), True)
degYUV = np.stack([degY, degU, degV], -1)[:numFrames]
degRGB = YUV2RGB(degYUV)/255

critic = ProxyNetwork().model()
# the generator model will be hrNet
generator = GeneratorBasic().model()
optimizerCritic = keras.optimizers.Adam(learning_rate=1e-3)
optimizerGenerator = keras.optimizers.Adam(learning_rate=1e-3)

EPOCHS = 100
LossFunc = keras.losses.MeanSquaredError()

train_metric_critic = keras.metrics.MeanSquaredError()
train_metric_gen_mse = keras.metrics.MeanSquaredError()
train_metric_gen_vmaf = keras.metrics.MeanSquaredError()

@tf.function
def train_step_gen(_x, _y, _yLuma):
    with tf.GradientTape() as tape:
        _genPred = generator([_x], training=True)
        _genPredLuma = tf.expand_dims(RGB2YUV_TF(_genPred)[:,:,:,0],-1)
        _criticPred = critic([_yLuma, _genPredLuma], training=False)
        _upper = tf.ones_like(_criticPred) * 100
        _genLoss = LossFunc(_criticPred, _upper) + LossFunc(_genPred, _y)
    grads = tape.gradient(_genLoss, generator.trainable_weights)
    optimizerGenerator.apply_gradients(zip(grads, generator.trainable_weights))
    return _genPred, _criticPred, _upper, _genPredLuma

@tf.function
def train_step_critic(_yLuma, _score, _genPredLuma):
    with tf.GradientTape() as tape:
        _pred = critic([_yLuma, _genPredLuma], training=True)
        _loss = LossFunc(_pred, _score)
    grads = tape.gradient(_loss, critic.trainable_weights)
    optimizerCritic.apply_gradients(zip(grads, critic.trainable_weights))
    return _pred

# DATAFRAME
# BATCHSIZE = 4
for _epoch in range(EPOCHS):
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