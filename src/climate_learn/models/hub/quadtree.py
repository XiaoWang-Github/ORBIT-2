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
    
    def set_area(self, mask, patch):
        # import pdb
        # pdb.set_trace()
        patch_size = self.get_size()
        # patch = np.resize(patch, patch_size)
        patch = patch.astype('float32')
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
        patch = np.expand_dims(patch, axis=-1)
        # import pdb
        # pdb.set_trace()
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
    
    def get_center(self):
        return (self.x2+self.x1)/2, (self.y2+self.y1)/2
    
    def draw(self, ax, c='grey', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
    
    def draw_area(self, ax, c='green', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)
    
    def draw_rescale(self, ax, c='green', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=16, 
                                 height=16, 
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)
    
    def draw_zorder(self, ax, c='red', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=16, 
                                 height=16, 
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)
    
                 
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
            if bbox.get_size()[0] == 2:
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
        seq_size = []
        seq_pos = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        h2,w2,c2 = size
        
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            #assert h1==w1, "Need squared input."
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
            # assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
        if len(seq_patch)<self.fixed_length:
            # import pdb
            # pdb.set_trace()
            if c2 > 1:
                seq_patch += [np.zeros(shape=(h2,w2,c2))] * (self.fixed_length-len(seq_patch))
            else:
                seq_patch += [np.zeros(shape=(h2,w2))] * (self.fixed_length-len(seq_patch))
            seq_size += [0]*(self.fixed_length-len(seq_size))
            seq_pos += [tuple([-1,-1])]*(self.fixed_length-len(seq_pos))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
        assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
        return seq_patch, seq_size, seq_pos
    
    def deserialize(self, seq, patch_size, channel):

        H,W = self.domain.shape
        seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, channel))
        seq = seq.astype(int)
        mask = np.zeros(shape=(H, W, channel))
        #print("demask:", mask.shape)
        
        # mask = np.expand_dims(mask, axis=-1)
        for idx,(bbox,value) in enumerate(self.nodes):
            pred_mask = seq[idx, ...]
            mask = bbox.set_area(mask, pred_mask)
        return mask
    
    def draw(self, ax, c='grey', lw=1):
        for bbox,value in self.nodes:
            bbox.draw(ax=ax)
    
    def draw_area(self, ax, c='green', lw=1):
        for bbox,value in self.nodes:
            bbox.draw_area(ax=ax, c=c, lw=lw)
            
    def draw_rescale(self, ax, c='green', lw=1):
        for bbox,value in self.nodes:
            bbox.draw_rescale(ax=ax, c=c, lw=lw)
            
    def draw_zorder(self, ax, c='red', lw=1):
        xs = []
        ys = []
        for bbox,value in self.nodes:
            x,y = bbox.get_center()
            xs += [x]
            ys += [y]
        ax.plot(xs, ys, color='red', linewidth=1)
        
class DensityQuadtree(FixedQuadTree):
    def __init__(self, domain, fixed_length=128, build_from_info=False, meta_info=None):
        super().__init__(domain, fixed_length, build_from_info, meta_info)
    
    def _build_tree(self):
        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        # self.nodes = [[root, root.contains(self.domain)]]
        r = root.contains(self.domain)/h/w
        m = root.contains(self.domain)
        self.nodes = [[root, m*r*r]]
        while len(self.nodes)<self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x:x[1])
            idx = self.nodes.index([bbox, value])
            if sum(bbox.get_size())<4:
                break

            x1,x2,y1,y2 = bbox.get_coord()
            lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
            m1 = lt.contains(self.domain)
            r1 = lt.contains(self.domain)/lt.get_size()/lt.get_size()
            v1 = m1*r1*r1
            
            rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
            v2 = rt.contains(self.domain)
            
            lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
            v3 = lb.contains(self.domain)
            
            rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
            v4 = rb.contains(self.domain)
            
            self.nodes = self.nodes[:idx] + [[lt,v1], [rt,v2], [lb,v3], [rb,v4]] +  self.nodes[idx+1:]

            # print([v for _,v in self.nodes])

        
    
                
