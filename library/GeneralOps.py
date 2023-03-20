import json 
import os 
import numpy as np
import subprocess 
import tensorflow as tf 

def saveModel(folderName, model, arguments):
    os.makedirs(folderName, exist_ok=True)
    model.save_weights(os.path.join(folderName, "model.h5"))
    with open(os.path.join(folderName, "arguments.json"), "w") as f:
        json.dump(arguments, f)

def readYUV420RangePatches(name: str, resolution: tuple, frameRange: tuple, patchLoc: tuple, patchSize: tuple, upsampleUV: bool = False):
    width = resolution[0]
    height = resolution[1]
    patchLoc_w = patchLoc[0]
    patchLoc_h = patchLoc[1]
    patchSize_w = patchSize[0]
    patchSize_h = patchSize[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    bytesYUV = bytesY + 2*bytesUV
    Y = []
    U = []
    V = []
    startLocation = frameRange[0]
    endLocation = frameRange[1] + 1
    for frameCnt in range(startLocation, endLocation, 1):
        startLocationBytes = (frameCnt) * (bytesYUV)
        YPatches = []
        UPatches = []
        VPatches = []
        for _row in range(patchSize_h):
            offSetBytesStartY = (patchLoc_h * width) + patchLoc_w + startLocationBytes + (_row*width)
            offSetBytesEndY = offSetBytesStartY + patchSize_w
            with open(name,"rb") as yuvFile:
                YPatches.append(np.fromfile(yuvFile, np.uint8, offSetBytesEndY-offSetBytesStartY, offset=offSetBytesStartY).reshape(patchSize_w))
        for _row in range(patchSize_h//2):
            offSetBytesStartU = startLocationBytes + bytesY + (patchLoc_h//2 * width//2) +  patchLoc_w//2 + (_row*(width//2))
            offSetBytesEndU = offSetBytesStartU + patchSize_w//2
            offSetBytesStartV = startLocationBytes + bytesY + bytesUV + (patchLoc_h//2 * width//2) +  patchLoc_w//2 + (_row*(width//2))
            offSetBytesEndV = offSetBytesStartV + patchSize_w//2
            with open(name,"rb") as yuvFile:
                UPatches.append(np.fromfile(yuvFile, np.uint8, offSetBytesEndU-offSetBytesStartU, offset=offSetBytesStartU).reshape(patchSize_w//2))
            with open(name,"rb") as yuvFile:
                VPatches.append(np.fromfile(yuvFile, np.uint8, offSetBytesEndV-offSetBytesStartV, offset=offSetBytesStartV).reshape(patchSize_w//2))
        YPatches = np.reshape(np.concatenate(YPatches,0), (patchSize_w,patchSize_h))
        UPatches = np.reshape(np.concatenate(UPatches,0), (patchSize_w//2,patchSize_h//2))
        VPatches = np.reshape(np.concatenate(VPatches,0), (patchSize_w//2,patchSize_h//2))
        Y.append(YPatches), U.append(UPatches), V.append(VPatches)
    Y = np.stack(Y, 0)
    U = np.stack(U, 0)
    V = np.stack(V, 0)
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V

def readYUV420(name: str, resolution: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        while (chunkBytes := yuvFile.read(bytesY + 2*bytesUV)):
            Y.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesY, offset = 0), (width, height)))
            U.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY),  (width//2, height//2)))
            V.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY + bytesUV), (width//2, height//2)))
    Y = np.stack(Y)
    U = np.stack(U)
    V = np.stack(V)
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V

def readYUV420Range(name: str, resolution: tuple, range: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        startLocation = range[0]
        endLocation = range[1] + 1
        startLocationBytes = startLocation * (bytesY + 2*bytesUV)
        endLocationBytes = endLocation * (bytesY + 2*bytesUV)
        data = np.fromfile(yuvFile, np.uint8, endLocationBytes-startLocationBytes, offset=startLocationBytes).reshape(-1,bytesY + 2*bytesUV)
        Y = np.reshape(data[:, :bytesY], (-1, width, height))
        U = np.reshape(data[:, bytesY:bytesY+bytesUV], (-1, width//2, height//2))
        V = np.reshape(data[:, bytesY+bytesUV:bytesY+2*bytesUV], (-1, width//2, height//2))
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V

def writeYUV420(name: str, Y, U, V, downsample=True):
    towrite = bytearray()
    if downsample:
        U = U[:, ::2, ::2]
        V = V[:, ::2, ::2]
    for i in range(Y.shape[0]):
        towrite.extend(Y[i].tobytes())
        towrite.extend(U[i].tobytes())
        towrite.extend(V[i].tobytes())
    with open(name, "wb") as destination:
        destination.write(towrite)

#input is an YUV numpy array with shape (batchSize,height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (batchSize,height,width,3), values in the range 0..255
def YUV2RGB(yuv):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,:,0]-=179.45477266423404
    rgb[:,:,:,1]+=135.45870971679688
    rgb[:,:,:,2]-=226.8183044444304
    rgb = np.clip(rgb,0,255)
    return rgb

#input is a RGB numpy array with shape (batchSize,height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (batchSize,height,width,3), values in the range 0..255
def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:,:,1:]+=128.0
    yuv = np.clip(yuv,0,255)
    return yuv

def RGB2YUV_TF(rgb):
    rgb = rgb*255.0
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    m = tf.convert_to_tensor(m,dtype=tf.dtypes.float32)
    yuv = tf.experimental.numpy.dot(rgb,m)
    offset_c0 = tf.zeros((tf.shape(yuv)[0], tf.shape(yuv)[1], tf.shape(yuv)[2], 1)) 
    offset_c12 = tf.ones((tf.shape(yuv)[0], tf.shape(yuv)[1], tf.shape(yuv)[2], 2)) * 128.0
    offset = tf.concat([offset_c0, offset_c12], -1)
    yuv = yuv + offset
    yuv = tf.clip_by_value(yuv,0,255)/255.0
    return yuv

# Function for running a command using the default user command line tool
def runTerminalCmd(command):
    process = subprocess.run(command, shell=True)


def upSample2XTile(x):
    _shape = x.shape
    y = np.empty((_shape[0], _shape[1]*2, _shape[2]*2, 3))
    y[:,0:_shape[1],0:_shape[2],:] = x
    y[:,_shape[1]:_shape[1]*2,0:_shape[2],:] = x
    y[:,0:_shape[1],_shape[2]:_shape[2]*2,:] = x
    y[:,_shape[1]:_shape[1]*2,_shape[2]:_shape[2]*2,:] = x
    return y