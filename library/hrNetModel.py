## Implementation of HrNet using layers in hrNetComponents

import tensorflow as tf
from keras import layers
from tensorflow import keras
from library.hrNetComponents import hrBlock, hrFinalBlock

class hrNet(keras.Model):
    def __init__(self, numLayersPerBlock=4, filters=[64,128,256,512], kernelSize=3, inputSize=192, outputSize=3):
        super().__init__()
        self.numLayersPerBlock = numLayersPerBlock
        self.filters = filters
        self.numSteps = len(self.filters)
        self.network = []
        self.kernelSize = kernelSize
        self.outputSize = outputSize
        self.inputSize = inputSize

        _inputSize = self.inputSize
        for i in range(self.numSteps):
            _numFilters = self.numSteps - i
            _interNetwork = []
            for _ in range(_numFilters):
                _interNetwork.append(hrBlock(self.filters[i], self.kernelSize, _inputSize, self.numLayersPerBlock))
            self.network.append(_interNetwork)
            _inputSize = _inputSize//2
        self.lastBlock = hrFinalBlock()

    def call(self, x, training=False):
        x_res = x[:,:,:,3:6]
        _processing = []
        _inputs = [x]
        for i in range(self.numSteps):
            for j in range(i,-1,-1):
                _processing.append(self.network[i-j][j](_inputs, training=training))
            _inputs = _processing[-(i+1):]


        outputs = _processing[-self.numSteps:]
        #Upsample Outputs
        for i in range(len(outputs)):
            
            _shape = outputs[i].shape
            _shape = (_shape[1],_shape[2])
            if _shape[0] < self.inputSize:
                _ratio = self.inputSize//_shape[0]
                outputs[i] = layers.UpSampling2D(_ratio)(outputs[i])


        outputs = tf.concat(outputs,-1)
        outputs = self.lastBlock(outputs, training=training)
        outputs = layers.Add()([outputs, x_res])

        return outputs
    
    def model(self):
        x = keras.Input(shape=(self.inputSize,self.inputSize,9))
        return keras.Model(inputs=[x], outputs=self.call(x))
    