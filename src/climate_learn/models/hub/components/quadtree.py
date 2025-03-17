import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def set_area(self, mask, patch, num_channels):
        patch_size = self.get_size()
        patch = patch.astype('float32')
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
        if num_channels <= 1:
            patch = np.expand_dims(patch, axis=-1)
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
    
    
class FixedQuadTree:
    def __init__(self, domain, fixed_length=128, build_from_info=False, meta_info=None) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        if build_from_info:
            self.nodes = self.decoder_nodes(meta_info=meta_info)
        else:
            self._build_tree()
    
    def nodes_value(self):
        meta_value = []
        for rect,v in self.nodes:
            size,_ = rect.get_size()
            meta_value += [[size/8]]
        return meta_value
    
    def encode_nodes(self):
        meta_info = []
        for rect,v in self.nodes:
            meta_info += [[rect.x1,rect.x2,rect.y1,rect.y2]]
        return meta_info
    
    def decoder_nodes(self, meta_info):
        nodes = []
        for info in meta_info:
            x1,x2,y1,y2 = info
            n = Rect(x1, x2, y1, y2)
            v = n.contains(self.domain)
            nodes +=  [[n,v]] 
        return nodes
            
    def _build_tree(self):
    
        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        self.nodes = [[root, root.contains(self.domain)]]
        while len(self.nodes)<self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x:x[1])
            idx = self.nodes.index([bbox, value])
            if bbox.get_size()[0] == 2 or bbox.get_size()[1] == 2:
                break

            x1,x2,y1,y2 = bbox.get_coord()
            lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
            v1 = lt.contains(self.domain)
            rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
            v2 = rt.contains(self.domain)
            lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
            v3 = lb.contains(self.domain)
            rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
            v4 = rb.contains(self.domain)
            
            self.nodes = self.nodes[:idx] + [[lt,v1], [rt,v2], [lb,v3], [rb,v4]] +  self.nodes[idx+1:]

            # print([v for _,v in self.nodes])
            
    def count_patches(self):
        return len(self.nodes)
    
    def serialize(self, img, size=(8,8,3)):
        
        seq_patch = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            
        h2,w2,c2 = size
        
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
        if len(seq_patch)<self.fixed_length:
            if c2 > 1:
                seq_patch += [np.zeros(shape=(h2,w2,c2))] * (self.fixed_length-len(seq_patch))
            else:
                seq_patch += [np.zeros(shape=(h2,w2))] * (self.fixed_length-len(seq_patch))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        return seq_patch
    
    def deserialize(self, seq, patch_size, channel):

        H,W = self.domain.shape
        seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, channel))
        seq = seq.astype(int)
        mask = np.zeros(shape=(H, W, channel))
        
        for idx,(bbox,value) in enumerate(self.nodes):
            pred_mask = seq[idx, ...]
            mask = bbox.set_area(mask, pred_mask, channel)
        return mask
