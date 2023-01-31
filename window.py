import numpy as np 
from skimage.util.shape import view_as_windows
import scipy 

# Function takes in three arguments: Input Array {HxWxC}, desired patch size
def deconstReconst(arr, patchSize=192, windowType='barthann'):
    inputHeight, inputWidth = arr.shape[0], arr.shape[1]

    #PAD HEIGHT AND WIDTH TO MULTIPLE OF PATCHSIZE
    nearestHeight = 0 
    nearestWidth = 0 
    cnt = 0 
    while nearestWidth < inputWidth:
        nearestWidth = cnt * patchSize
        cnt += 1 
    cnt = 0 
    while nearestHeight < inputHeight:
        nearestHeight = cnt * patchSize
        cnt += 1 
    toPadX, toPadY = ((nearestWidth - inputWidth + patchSize)//2), (nearestHeight - inputHeight + patchSize//2)
    arr = np.pad(arr, ((toPadY, toPadY),(toPadX, toPadX), (0,0)), 'edge')
    arr = view_as_windows(arr, (patchSize, patchSize, 3), step=patchSize//2)
    reformed_shape = arr.shape 
    arr = np.reshape(arr, (-1, patchSize, patchSize, 3))
    # Variable "arr" are N * patchSize * patchSize * 3 patches with some overlap 
    # PROCESS PATCHES BELOW
    # # # # # 

    arr = np.clip(np.random.normal(0,0.25,size=arr.shape) + arr, 0, 1)

    # # # # # 
    if windowType == 'bartlett':
        window1d = scipy.signal.windows.bartlett(patchSize)
    elif windowType == 'blackman':
        window1d = scipy.signal.windows.bartlett(patchSize)
    if windowType == 'barthann':
        window1d = scipy.signal.windows.barthann(patchSize)
    window2d = np.reshape((np.outer(window1d, window1d)),(patchSize, patchSize, 1))
    arr = arr * window2d
    arr = np.reshape(arr, reformed_shape)
    outputArr = np.zeros((inputHeight+2*toPadY, inputWidth+2*toPadX, 3))
    for y in range(reformed_shape[0]):
        for x in range(reformed_shape[1]):
            currentSlice = np.s_[y*patchSize//2:(y*patchSize//2) + patchSize, x*patchSize//2:(x*patchSize//2) + patchSize, :]
            outputArr[currentSlice] = outputArr[currentSlice] + arr[y, x, 0]
    outputArr = outputArr[toPadY:-toPadY, toPadX:-toPadX, :]
    return outputArr


from skimage import data
import matplotlib.pyplot as plt

img = data.astronaut()
img = img/255.0 
out_img = deconstReconst(img, 192)

plt.imshow(img)
plt.title("INPUT IMAGE")
plt.show()

plt.imshow(out_img)
plt.title("OUTPUT IMAGE")
plt.show()