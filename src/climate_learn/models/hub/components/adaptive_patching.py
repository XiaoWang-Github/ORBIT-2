import numpy as np
import cv2 as cv
import torch
import scipy
from .quadtree import FixedQuadTree
class Patchify(torch.nn.Module):
    def __init__(self, fixed_length=196, patch_size=16, num_channels=3) -> None:
        super().__init__()
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels

    def forward(self, img, edges):  # we assume inputs are always structured like this
        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img= qdt.serialize(torch.moveaxis(img,0,-1), size=(self.patch_size,self.patch_size,self.num_channels))
        seq_img = torch.stack(seq_img)

        if self.num_channels > 1:
            seq_img = torch.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size])
        else:
            seq_img = torch.reshape(seq_img, [-1, self.patch_size*self.patch_size])

        return seq_img, qdt
