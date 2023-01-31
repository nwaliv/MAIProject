## Custom Layers that create HrNet

import tensorflow as tf
from keras import layers
from tensorflow import keras
import tensorflow_addons as tfa

class CAModule(layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.numFilters = numFilters

        self.MaxPool = layers.GlobalMaxPooling2D(data_format='channels_last')
        self.AvgPool = layers.GlobalAveragePooling2D(data_format='channels_last')

        self.denseUnits = [self.numFilters,self.numFilters, self.numFilters/2, self.numFilters/4, self.numFilters/2, self.numFilters,self.numFilters]
        self.denseLayers = []
        for unit in self.denseUnits:
            self.denseLayers.append(layers.Dense(unit, activation=keras.activations.sigmoid))
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters
        })
        return config
    
    def call(self, inputs, training=False):
        maxPool = self.MaxPool(inputs, training=training)
        maxPool = layers.Reshape((1,1,maxPool.shape[-1]))(maxPool)

        avgPool = self.AvgPool(inputs, training=training)
        avgPool = layers.Reshape((1,1,avgPool.shape[-1]))(avgPool)

        for denseLayer in self.denseLayers:
            maxPool = denseLayer(maxPool, training=training)
        for denseLayer in self.denseLayers:
            avgPool = denseLayer(avgPool, training=training)
        combinedPool = maxPool + avgPool
        x = keras.activations.softmax(combinedPool)
        return x

class SAModule(layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.numFilters = numFilters
        self.cnnFilters = [4,2,2,1]
        self.CNNLayers = []
        for filter in self.cnnFilters:
            self.CNNLayers.append(layers.Conv2D(filter,3,padding='SAME', activation=keras.activations.sigmoid))
        
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters
        })
        return config
    
    def call(self, inputs, training=False):
        maxPool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        avgPool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        combinedPool = layers.Concatenate()([maxPool, avgPool])
        for layer in self.CNNLayers:
            combinedPool = layer(combinedPool, training=training)
        x = keras.activations.softmax(combinedPool)
        return x



class CBAModule(layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.numFilters = numFilters
        self.CAModule = CAModule(self.numFilters)
        self.SAModule = SAModule(self.numFilters)

    def get_config(self):
        config = super().get_config().copy()
        config.update({})
        return config
    
    def call(self, inputs, training=False):
        xCA = self.CAModule(inputs, training=training)
        xCA = tf.broadcast_to(xCA,tf.shape(inputs))
        xCA = layers.Multiply()([xCA, inputs])
        xSA = self.SAModule(xCA, training=training)
        xSA = tf.broadcast_to(xSA, tf.shape(xCA))
        x = layers.Multiply()([xSA, xCA])
        return x

class hrLayers(layers.Layer):
    def __init__(self, numFilters, size, strides=(1, 1), padding='same', **kwargs):
        super().__init__()
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.conv1 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu1_2 = layers.Activation(tf.nn.elu)
        self.conv2 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu1 = layers.Activation(tf.nn.elu)
        self.conv3 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu3_4 = layers.Activation(tf.nn.elu)
        self.conv4 = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.relu2 = layers.Activation(tf.nn.elu)
        self.convP = layers.Conv2D(self.numFilters, self.size, self.strides, self.padding, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.groupNorm = tfa.layers.GroupNormalization(16)
        self.add = layers.Add()
        self.outputRelu = layers.Activation(tf.nn.elu)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'strides' : self.strides,
            'padding' : self.padding
        })
        return config
    
    def call(self, inputs, training=False):
        x_0 = self.conv1(inputs,training=training)
        x_0 = self.relu1_2(x_0, training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv3(x_0,training=training)
        x_0 = self.relu3_4(x_0,training=training)
        x_0 = self.conv4(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.groupNorm(x_1,training=training)
        x = self.add([x_0, x_1], training=training)
        x = self.outputRelu(x,training=training)
        return x


class hrBlock(layers.Layer):
    def __init__(self, numFilters, kernelSize, inputSize, numLayersPerBlock=6):
        super().__init__()
        self.numFilters = numFilters
        self.inputSize = inputSize
        self.numLayersPerBlock = numLayersPerBlock
        self.kernelSize = kernelSize
        self.block = []
        for i in range(self.numLayersPerBlock):
            # if i == 1 :
            #     self.block.append([CBAModule(self.numFilters), hrLayers(self.numFilters, self.kernelSize)])
            # else:
            self.block.append(hrLayers(self.numFilters, self.kernelSize))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'inputSize' : self.inputSize,
            'kernelSize': self.kernelSize
        })
        return config
    
    def call(self, inputs, training=False):
        # Tensor shape is (batchSize, height, width, channels)
        for i in range(len(inputs)):
            # Check width/height
            _shape = inputs[i].shape
            _shape = (_shape[1],_shape[2])
            if _shape[0] != _shape[1]:
                raise TypeError(f"Internal height({_shape[0]}) and width({_shape[1]}) are not equal")    
            # Downsample
            if _shape[0] > self.inputSize:
                _ratio = _shape[0]//self.inputSize
                inputs[i] = layers.AveragePooling2D(_ratio)(inputs[i])
            # Upsample 
            if _shape[0] < self.inputSize:
                _ratio = self.inputSize//_shape[0]
                inputs[i] = layers.UpSampling2D(_ratio)(inputs[i])
        
        x_0 = tf.concat(inputs,-1)
        for i in range(len(self.block)):
            # if i == 1:
            #     x_cba = self.block[i][0](x_0, training = training)
            #     x_0 = self.block[i][1](x_0, training = training)
            #     x_0 = layers.Multiply()([x_cba, x_0])
            # else:
            x_0 = self.block[i](x_0, training=training)
        return x_0
    

class hrFinalBlock(layers.Layer):
    def __init__(self, filters=[512,128,64,32,16,9,3], kernelSize=5):
        super().__init__()
        self.filters = filters
        self.Conv2D = []
        self.kernelSize = kernelSize

        for filter in self.filters:
            self.Conv2D.append(layers.Conv2D(filter, self.kernelSize, padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform()))
            self.Conv2D.append(layers.Activation(tf.nn.elu))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernelSize' : self.kernelSize
        })
        return config

    def call(self, x, training=False):
        for filter in self.Conv2D:
            x = filter(x, training=training)
        return x