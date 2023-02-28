import tensorflow as tf 
from tensorflow import keras 
import tensorflow_addons as tfa 
import os 

class Critic(keras.Model):
    def __init__(self, dsFiltersUnet=[32,64,84,128], centerFiltersUnet=[162,162], numCnnsUnet=2, outputFilters=[32,16,1], effNet=False):
        super(Critic, self).__init__()
        self.UNET = UNET(dsFiltersUnet, centerFiltersUnet, numCnnsUnet, normalization=True)
        self.effNet = effNet
        if self.effNet:
            self.effLayer = keras.models.load_model(os.path.join(os.getcwd(), "data/models/effNetOptimOut.h5"))
            for _l in self.effLayer.layers:
                _l.trainable = False
        self.outputCNN = []
        for _filter in outputFilters[:-1]:
            self.outputCNN.append(cnnOpPaddedSpecNorm(_filter, 5))
        self.outputCNN.append(cnnOpPaddedSpecNorm(outputFilters[-1], 5, activation='linear'))

    def call(self, xRGB, training=False):
        if self.effNet:
            x_effNet = self.effLayer(xRGB*255)
            x = self.UNET(xRGB, x_effNet, training=training)
        else:
            x = self.UNET(xRGB, training=training)
        for _layer in self.outputCNN:
            x = _layer(x, training=training)      
        return x

    def model(self):
        xRGB = keras.Input(shape=(None, None, 3), name='Restored Frame (RGB)')
        return keras.Model(inputs=[xRGB], outputs=self.call(xRGB))

class ProxyNetwork(keras.Model):
    def __init__(self, dsFilters=[8,16,32,64,72,86], concatFilters=[128,128], numCnnsBlock=1, numCnnsConcat=1, denseUnits=[128,64,32,16,1]):
        super(ProxyNetwork, self).__init__()
        self.dsConv2dLayersRef = []
        self.dsConv2dLayersDeg = []
        for _filter in dsFilters:
            for _ in range(numCnnsBlock - 1):
                self.dsConv2dLayersRef.append(cnnOpPaddedSpecNorm(_filter,3,paddingAmt=1))
                self.dsConv2dLayersDeg.append(cnnOpPaddedSpecNorm(_filter,3,paddingAmt=1))
            self.dsConv2dLayersRef.append(cnnOpPaddedSpecNorm(_filter,5, strides=2))
            self.dsConv2dLayersDeg.append(cnnOpPaddedSpecNorm(_filter,5, strides=2))
        self.concatConv2dLayers = []
        for _filter in concatFilters:
            for _ in range(numCnnsConcat-1):
                self.concatConv2dLayers.append(cnnOpPaddedSpecNorm(_filter, 3, paddingAmt=1))
            self.concatConv2dLayers.append(cnnOpPaddedSpecNorm(_filter,5, strides=2))
        self.flattenLayer = keras.layers.Flatten()
        self.denseLayers = []
        for _filter in denseUnits[:-1]:
            self.denseLayers.append(keras.layers.Dense(_filter, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        self.denseLayers.append(keras.layers.Dense(denseUnits[-1], activation='linear'))

    def call(self, xref, xdeg, training=False):
        for _layer in self.dsConv2dLayersRef:
            xref = _layer(xref, training=training)
        for _layer in self.dsConv2dLayersDeg:
            xdeg = _layer(xdeg, training=training)
        x = tf.concat([xref, xdeg], -1)
        for _layer in self.concatConv2dLayers:
            x = _layer(x, training=training)
        x = self.flattenLayer(x, training=training)
        for _layer in self.denseLayers:
            x = _layer(x, training=training)
        return x

    def model(self):
        # Changed input shape from 512 to 192
        xref = keras.Input(shape=(192, 192, 1), name='Ref (Y)')
        xdeg = keras.Input(shape=(192, 192, 1), name='Deg (Y)')
        return keras.Model(inputs=[xref, xdeg], outputs=self.call(xref, xdeg))

class UNET(keras.layers.Layer):
    def __init__(self, dsFilters=[32,64,84,128], centerFilters=[136,136], numCNNs=2, normalization=False):
        super(UNET, self).__init__()
        self.normalization = normalization
        self.downBlocks = []
        self.skipConnectIdx = numCNNs
        for _filter in dsFilters:
            for _ in range(numCNNs-1):
                self.downBlocks.extend([cnnOpPaddedSpecNorm(_filter, 3, paddingAmt=1)])
            self.downBlocks.extend([cnnOpPaddedSpecNorm(_filter, 5, paddingAmt=2, strides=2, normalization=normalization)])
        self.centerBlocks = []
        for _filter in centerFilters:
            self.centerBlocks.extend([cnnOpPaddedSpecNorm(_filter, 3, paddingAmt=1, normalization=normalization)])
        upFilters = reversed(dsFilters)
        self.upBlocks = []
        for _filter in upFilters:
            self.upBlocks.append(keras.layers.UpSampling2D(interpolation='bilinear'))
            for _ in range(numCNNs-1):
                self.upBlocks.extend([cnnOpPaddedSpecNorm(_filter, 3, paddingAmt=1)])
            self.upBlocks.extend([cnnOpPaddedSpecNorm(_filter, 3, paddingAmt=1, normalization=normalization)])

    def call(self, x, x_EffNet=None, training=False):
        skipConnections = []
        if x_EffNet:
            x_EffNet.reverse()
        for idx, _layer in enumerate(self.downBlocks):
            x = _layer(x, training=training)
            if (idx % self.skipConnectIdx == (self.skipConnectIdx-1)):
                skipConnections.append(x)
            if x_EffNet:
                if idx in [1*self.skipConnectIdx-1, 4*self.skipConnectIdx-1]:
                    x = tf.concat([x, x_EffNet.pop()], -1)
        for _layer in self.centerBlocks:
            x = _layer(x, training=training)
        for idx, _layer in enumerate(self.upBlocks):
            if (idx % (self.skipConnectIdx+1) == 0):
                x = tf.concat([x, skipConnections.pop()], -1)
            x = _layer(x, training=training)
        return x 


class cnnOpPaddedSpecNorm(keras.layers.Layer):
    def __init__(self, filter=32,ksize=5, padding='VALID', activation=tf.keras.layers.LeakyReLU(alpha=0.3), paddingAmt=2, strides=1, normalization=False):
        super(cnnOpPaddedSpecNorm, self).__init__()
        self.paddingAmt = paddingAmt
        self.normalization = normalization
        if self.normalization:
            self.conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(filter, ksize, padding=padding, activation=activation, strides=strides))
        else:
            self.conv2d = keras.layers.Conv2D(filter, ksize, padding=padding, activation=activation, strides=strides)
    
    def call(self, x, training=False):
        x = tf.pad(x, [[0, 0], [self.paddingAmt, self.paddingAmt],[self.paddingAmt, self.paddingAmt], [0, 0]], mode='SYMMETRIC')
        x = self.conv2d(x, training=training)
        return x