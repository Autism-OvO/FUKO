import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layers import *
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x ,self.ic , self.oc

class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)
    