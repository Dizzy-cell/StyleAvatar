
import math

import torch
from torch import nn

from networks.modules import *

from IPython import embed 
import time 

class FaceUNet(nn.Module):
    def __init__(
        self,
        input_size = 128,
        output_size = 512,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.in_log_size = int(math.log(input_size, 2)) - 1
        self.out_log_size = int(math.log(output_size, 2)) - 1
        self.comb_num = self.in_log_size - 5

        # add new layer here
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()

        in_channel = self.channels[self.input_size]
        for i in range(self.in_log_size - 2, 2, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs.append(FromRGB(in_channel, 3, downsample=True))
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel = blur_kernel))
            in_channel = out_channel

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()
        
        in_channel = self.channels[8]

        for i in range(4, self.out_log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(UPConv(in_channel, out_channel,  kernel_size = 3, upsample = True, blur_kernel = blur_kernel))
            if i - 4 < self.comb_num:
                self.comb_convs.append(ConvBlock(out_channel * 2, out_channel, blur_kernel = blur_kernel, downsample = False))
            self.to_rgbs.append(ToRGB_nostyle(out_channel))

            in_channel = out_channel

        self.iwt = InverseHaarTransform(3)

    def forward(self, condition_img):

        #latent = torch.zeros((condition_img.shape[0], self.style_dim)).to(style.device)

        cond_img = self.dwt(condition_img) #(Bx12x256x256)

        cond_out = None
        cond_list = []
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            cond_list.append(cond_out)

        #cond_img -> (Bx12x16x16)

        i = 0
        skip = None
        out = cond_list[self.comb_num]
        for to_rgb, conv in zip(self.to_rgbs, self.convs):

            out = conv(out)
            skip = to_rgb(out, skip)

            i += 1
            if i <= self.comb_num: 
                out = torch.cat([out, cond_list[self.comb_num - i]], dim = 1)
                out = self.comb_convs[i-1](out)

        image = self.iwt(skip)

        return image
