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
alpha = [2.5e-06]
for _alpha in alpha:
    modelName = f'HRNET_CAMBI_ALPHA{(_alpha)}_FINAL'
    modelNameCritic = f'PROXY_CAMBI_ALPHA{(_alpha)}_FINAL'

    N_CRITICS = 2

    outputJson = f'results/{modelName}.json'
    # the generator model will be hrNet
    generator = hrNet(2, [32, 64, 128, 256], 5).model()
    # Read in the proxy network for CAMBI
    critic = ProxyNetwork().model()
    generator.load_weights('results/HRNET_MSE.h5')
    critic.load_weights('results/CRITIC_MSE.h5')

    # Define the optimizers and loss function
    optimizerGenerator = keras.optimizers.RMSprop(learning_rate=1e-5)
    optimizerCritic = keras.optimizers.RMSprop(learning_rate=1e-5)
    EPOCHS = 10
    LossFunc = keras.losses.MeanSquaredError()
    train_metric_gen_mse = keras.metrics.MeanSquaredError()
    train_metric_gen_cambi = keras.metrics.Mean()
    train_metric_critic_mse = keras.metrics.MeanSquaredError()


    print(f"GPUS: {tf.config.list_physical_devices('GPU')}")

    # Define the training functions
    @tf.function
    def train_step_gen(_x, _y, _yLuma, alpha):
        # _x is the degraded image, _y is the reference image, _yLuma is the luma of the reference image
        with tf.GradientTape() as tape:
            # Generate the prediction
            _genPred = generator([_x], training=True)
            # Get the luma of the prediction
            _genPredLuma = tf.expand_dims(RGB2YUV_TF(_genPred)[:,:,:,0],-1)
            # Get the critic score for the prediction
            _criticPred = critic([_yLuma, _genPredLuma], training=False)
            # Define the loss function for the generator as the sum of the critic score and the MSE
            _genLoss = (alpha)*_criticPred + (LossFunc(_genPred, _y))
        # Apply the gradients to the generator
        grads = tape.gradient(_genLoss, generator.trainable_weights)
        optimizerGenerator.apply_gradients(zip(grads, generator.trainable_weights))
        return _genPred, _genPredLuma, _criticPred

    @tf.function
    def train_step_critic(_genPredLuma, _yLuma, _score):
        for _ in range(N_CRITICS):
            # _yLuma is the luma of the reference image, _score is the critic score for the prediction, _genPredLuma is the luma of the prediction
            with tf.GradientTape() as tape:
                # Get the critic score for the prediction
                _pred = critic([_yLuma, _genPredLuma], training=True)
                # _pred = returnMetric
                # Define the loss function for the critic as the MSE between the critic score and the prediction
                _loss = LossFunc(_pred, _score)
            # Apply the gradients to the critic
            grads = tape.gradient(_loss, critic.trainable_weights)
            optimizerCritic.apply_gradients(zip(grads, critic.trainable_weights))
        return _pred

    # DATAFRAME
    BATCHSIZE = 4
    # read in the training data
    DATAFRAME = pd.read_csv('dataFrames/trainVideoSet_perframeCambi_maxFrame.csv')
    DATAFRAME = DATAFRAME[DATAFRAME['FrameNum'] != 0]
    DATAFRAME = DATAFRAME[DATAFRAME['FrameNum'] != DATAFRAME['MaxFrame']]
    DATAFRAME = DATAFRAME.reset_index(drop=True)
    
    baseRefPath = '/data/nwaliv/trainVideoSetRef/'
    baseCompPath = '/data/nwaliv/trainVideoSetDeg/'

    Loss = { 'MSE_GEN_Loss' : [],  'Cambi_GEN_Loss' : [], 'MSE_CRITIC_Loss' : []}

    # the training loop
    for _epoch in range(EPOCHS):
        # shuffle the dataframe
        indexes = np.arange(0, DATAFRAME.shape[0], 1)
        np.random.shuffle(indexes)
        _df = DATAFRAME.iloc[indexes].copy()
        # get the number of steps
        numSteps = _df.shape[0]//BATCHSIZE
        print(numSteps)
        for step in range(numSteps): 
            # get the batch
            SAMPLE = _df.sample(n=BATCHSIZE, replace=False)
            _df = _df.drop(index=SAMPLE.index.tolist())
            SAMPLE = SAMPLE.reset_index(drop=True)
            # initialize the input and target arrays
            X_in = np.zeros((BATCHSIZE, patchSize, patchSize, 9))
            Y_target = np.zeros((BATCHSIZE, patchSize, patchSize, 3))
            Y_target_luma = np.zeros((BATCHSIZE, patchSize, patchSize, 1))
            scores = np.zeros((BATCHSIZE))
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

                _yY = _yY/255.0
                # if random.random() < 0.50:
                #     _channels = [0, 1, 2]
                #     random.shuffle(_channels)
                #     _xRGB = _xRGB[...,_channels]
                #     _yRGB = _yRGB[...,_channels]

                _xRGB = np.concatenate((_xRGB[0], _xRGB[1], _xRGB[2]), axis=-1).reshape(1,patchSize,patchSize,9)

                # add the data to the arrays
                X_in[idx] = _xRGB[0]
                Y_target[idx] = _yRGB[0]
                Y_target_luma[idx] = np.expand_dims(_yY[0],-1)
            X_in, Y_target, Y_target_luma = X_in.astype(np.float32), Y_target.astype(np.float32), Y_target_luma.astype(np.float32)

            # train the generator
            _alpha = tf.convert_to_tensor(_alpha)
            _genPred, _genPredLuma, _genCambiScore = train_step_gen(X_in, Y_target, Y_target_luma, _alpha)

            _genPredTiled, Y_targetTiled = upSample2XTile(tf.clip_by_value(_genPred, 0.0, 1.0)), upSample2XTile(Y_target)
            _proxyTarget = returnMetric(_genPredTiled, Y_targetTiled, height=patchSize*2, width=patchSize*2).astype(np.float32)
            _criticPred = train_step_critic(_genPredLuma, Y_target_luma, _proxyTarget)
            
            train_metric_gen_mse.update_state(_genPred, Y_target)
            train_metric_gen_cambi.update_state(_genCambiScore)
            train_metric_critic_mse.update_state(_criticPred, _proxyTarget)

            if step % 1000 == 0:
                print(f'Epoch: {_epoch}, Step: {step}, Gen MSE: {train_metric_gen_mse.result()}, CAMBI Preds: {train_metric_gen_cambi.result()}, CAMBI MSE: {train_metric_critic_mse.result()}')
        Loss['MSE_GEN_Loss'].append(float(train_metric_gen_mse.result()))
        Loss['Cambi_GEN_Loss'].append(float(train_metric_gen_cambi.result()))
        Loss['MSE_CRITIC_Loss'].append(float(train_metric_critic_mse.result()))

        train_metric_gen_mse.reset_states()
        train_metric_gen_cambi.reset_states()
        train_metric_critic_mse.reset_states()

    generator.save_weights(f'results/{modelName}.h5')
    critic.save_weights(f'results/{modelNameCritic}.h5')

    with open(outputJson, "w") as outfile:
        json.dump(Loss, outfile)