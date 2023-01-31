# To Set up the LPIPS metric
import lpips
import torch
import numpy as np
from src.auxFunctions import image_preprocess

class lpipsClass():
    def __init__(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    def lpipsLossFunc(self, x, y):
        x, y = torch.tensor(x), torch.tensor(y)
        d = self.loss_fn_vgg(x, y)
        return d

    def computeLpipsFrame(self,restored_frame,original_frame):
        img0 = image_preprocess(restored_frame)
        img0 = np.expand_dims(img0,axis=0)
        img0 = np.moveaxis(img0,-1,1)
        img0 = np.float32(img0)
        # the restored frame
        img1 = image_preprocess(original_frame)
        img1 = np.expand_dims(img1,axis=0)
        img1 = np.moveaxis(img1,-1,1)
        img1 = np.float32(img1)
        # Converting from numpy to pytorch
        img0 = torch.tensor(img0)
        img1 = torch.tensor(img1)

        return self.lpipsLossFunc(img0,img1)
