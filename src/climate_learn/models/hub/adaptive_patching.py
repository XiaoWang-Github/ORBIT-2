import numpy as np
import cv2 as cv
import torch
import random
from .quadtree import FixedQuadTree

class Patchify(torch.nn.Module):
    #TODO: Pass dtype for preferred return dtype
    def __init__(self, sths=[1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3) -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        
    def forward(self, img):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        if self.smooth_factor ==0 :
            edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
        else:
            grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
            #edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
            edges = cv.Canny((grey_img*255).astype(np.uint8), self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.patch_size,self.patch_size,self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)
        #print("SEQ_IMG_SHAPE", seq_img.shape,flush=True)

        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size])

        seq_pos = np.asarray(seq_pos)
        #return seq_img, qdt, seq_size, seq_pos
        return seq_img, qdt
