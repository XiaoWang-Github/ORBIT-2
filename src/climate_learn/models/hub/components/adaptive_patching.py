import numpy as np
import cv2 as cv
import torch
import random
import scipy
from .quadtree import FixedQuadTree

class Patchify(torch.nn.Module):
    #TODO: Pass dtype for preferred return dtype
    def __init__(self, sths=[1,3,5], fixed_length=196, cannys=[50, 100], canny_add=50, patch_size=16, num_channels=3, physics=False, edge_percentage=.1, grad_deg=1) -> None:
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.canny_add = canny_add
        self.physics = physics
        self.edge_percentage = edge_percentage
        self.grad_deg = grad_deg

    def forward(self, img):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input

        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+self.canny_add]
        if self.physics:
            if self.grad_deg == 1:
                grad = np.gradient(np.squeeze(img))
                grad_out = np.sqrt(grad[0]**2 + grad[1]**2)
            else:
                grad = np.gradient(np.squeeze(img))
                grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
                grad_x = np.gradient(grad[0]/grad_mag, axis=0)
                grad_y = np.gradient(grad[1]/grad_mag, axis=1)
                grad_out = grad_x + grad_y

            ind = np.unravel_index(np.argsort(grad_out, axis=None), grad_out.shape)
            edges = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            topK = int(img.shape[0]*img.shape[1]*(self.edge_percentage-self.edge_percentage*.5))
            for i in range(img.shape[0]*img.shape[1]-topK, img.shape[0]*img.shape[1]):
                edges[ind[0][i],ind[1][i]] = 255
            for i in range(topK):
        else:
            if self.smooth_factor ==0 :
                edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            else:
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                #edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
                edges = cv.Canny((grey_img*255).astype(np.uint8), self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img = qdt.serialize(img, size=(self.patch_size,self.patch_size,self.num_channels))
        seq_img = np.asarray(seq_img, dtype=np.float32)

        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size])

        return seq_img, qdt
