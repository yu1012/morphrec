"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=512, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(self, img_size=2250, num_blocks=5, num_lead_in=2, num_lead_out=10, init_dim=64, style_dim=512, max_conv_dim=512):
        super().__init__()
        self.num_lead_in = num_lead_in
        self.num_lead_out = num_lead_out
        self.init_dim = init_dim
        self.img_size = img_size
        self.from_rgb = nn.Conv1d(self.num_lead_in, self.init_dim, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm1d(self.init_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.init_dim, self.num_lead_out, 1, 1, 0))

        for _ in range(num_blocks):
            dim_out = min(init_dim*2, max_conv_dim)
            self.encode.append(
                ResBlk(init_dim, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, init_dim, style_dim * self.num_lead_out, upsample=True))  # stack-like
            init_dim = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim * self.num_lead_out))

    def forward(self, x, s):
        x = self.from_rgb(x)
        h_intermediate = []
        for block in self.encode:
            x = block(x)
            h_intermediate.append(x)
        for block in self.decode:
            h = h_intermediate.pop()
            if x.shape[2] != h.shape[2]:
                x = F.interpolate(x, size=h.shape[2], mode='nearest')
            x = block(x, s)
        return self.to_rgb(x)


class StyleEncoder(nn.Module):
    def __init__(self, img_size=2250, num_blocks=5, num_lead_in=2, num_lead_out=10, init_dim=64, style_dim=512, max_conv_dim=512):
        super().__init__()
        initial_dim = init_dim
        self.num_lead_out = num_lead_out

        blocks = []
        blocks += [nn.Conv1d(num_lead_in, init_dim, 3, 1, 1)]

        for _ in range(num_blocks):
            dim_out = min(init_dim*2, max_conv_dim)
            blocks += [ResBlk(init_dim, dim_out, downsample=True)]
            init_dim = dim_out

        last_kernel_size = img_size // 2**num_blocks 
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv1d(dim_out, dim_out, last_kernel_size, 1, 0)] # 4
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = []
        for i in range(self.num_lead_out):
            self.unshared += [nn.Linear(dim_out, style_dim*self.num_lead_out)]
        self.lead_style_encoder = nn.ModuleList(self.unshared)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.lead_style_encoder:
            out += [layer(h)]
        out = torch.stack(out, dim=1).reshape(x.size(0), self.num_lead_out, self.num_lead_out, -1)  # (batch, num_domains, style_dim)
        idx = torch.arange(0, self.num_lead_out).to(x.device)
        out = out[:,idx,idx,:]
        s = out.reshape(x.shape[0], -1)
        return s


class StyleUNet(nn.Module):
    def __init__(self, img_size=2250, num_blocks=5, num_lead_in=2, num_lead_out=8, latent_dim=16, style_dim=512, max_conv_dim=512):
        super().__init__()
        self.num_lead_out = num_lead_out
        self.generator = Generator(img_size, num_blocks, num_lead_in, num_lead_out, style_dim=style_dim, max_conv_dim=max_conv_dim)
        self.style_encoder = StyleEncoder(img_size, num_blocks, num_lead_in, num_lead_out, style_dim=style_dim, max_conv_dim=max_conv_dim)
        
    def forward(self, x):
        style_embedding = self.style_encoder(x)
        output = self.generator(x, style_embedding)
        return output

if __name__=="__main__":
    model = StyleUNet()
    x = torch.randn(1, 2, 2250)
    outputs = model(x)
    print(outputs.shape)
