import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y

def upsample(x, scale_factor=2, mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5):
        super().__init__()

        self.den_dropout = den_dropout

        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.enc1 = nn.Sequential(*list(mobilenet.features.children())[:6])  # Output: [Batch, 40, H/2, W/2]
        self.enc2 = nn.Sequential(*list(mobilenet.features.children())[6:12])  # Output: [Batch, 112, H/4, W/4]
        self.enc3 = nn.Sequential(*list(mobilenet.features.children())[12:])  # Output: [Batch, 960, H/8, W/8]

        self.dec3 = nn.Sequential(
            ConvBlock(960, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(112 + 256, 256, bn=True),  # Corrected channels
            ConvBlock(256, 128, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(40 + 128, 128, bn=True),  # Corrected channels
            ConvBlock(128, 64, bn=True)
        )

        self.den_dec = nn.Sequential(
            ConvBlock(256 + 128 + 64, 128, kernel_size=1, padding=0, bn=True),  # Corrected channels
            nn.Dropout2d(p=den_dropout)
        )

        self.den_head = nn.Sequential(
            ConvBlock(128, 1, kernel_size=1, padding=0)
        )

    def pad_to_match(self, tensors):
        max_h = max(tensor.size(2) for tensor in tensors)
        max_w = max(tensor.size(3) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            h, w = tensor.size(2), tensor.size(3)
            pad_h = (max_h - h) // 2
            pad_w = (max_w - w) // 2
            pad = (pad_w, max_w - w - pad_w, pad_h, max_h - h - pad_h)
            padded_tensors.append(F.pad(tensor, pad))
        return padded_tensors

    def forward_fe(self, x):
        x1 = self.enc1(x)  # Output: [Batch, 40, H/2, W/2]
        x2 = self.enc2(x1)  # Output: [Batch, 112, H/4, W/4]
        x3 = self.enc3(x2)  # Output: [Batch, 960, H/8, W/8]

        x = self.dec3(x3)  # Output: [Batch, 256, H/8, W/8]
        y3 = x
        x = upsample(x, scale_factor=2)  # Output: [Batch, 256, H/4, W/4]
        x = torch.cat(self.pad_to_match([x, x2]), dim=1)  # Output: [Batch, 256 + 112, H/4, W/4]

        x = self.dec2(x)  # Output: [Batch, 128, H/4, W/4]
        y2 = x
        x = upsample(x, scale_factor=2)  # Output: [Batch, 128, H/2, W/2]
        x = torch.cat(self.pad_to_match([x, x1]), dim=1)  # Output: [Batch, 128 + 40, H/2, W/2]

        x = self.dec1(x)  # Output: [Batch, 64, H/2, W/2]
        y1 = x

        y2 = upsample(y2, scale_factor=2)  # Output: [Batch, 128, H, W]
        y3 = upsample(y3, scale_factor=4)  # Output: [Batch, 256, H, W]

        y1, y2, y3 = self.pad_to_match([y1, y2, y3])

        y_cat = torch.cat([y1, y2, y3], dim=1)  # Output: [Batch, 64 + 128 + 256, H, W]

        return y_cat, x3

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        d = self.den_head(y_den)
        d = upsample(d, scale_factor=4)

        return d

class DGModel_mem(DGModel_base):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5):
        super().__init__(pretrained, den_dropout)

        self.mem_size = mem_size
        self.mem_dim = mem_dim

        self.mem = nn.Parameter(torch.FloatTensor(1, self.mem_dim, self.mem_size).normal_(0.0, 1.0))

        self.den_dec = nn.Sequential(
            ConvBlock(256 + 128 + 64, self.mem_dim, kernel_size=1, padding=0, bn=True),  # Corrected channels
            nn.Dropout2d(p=den_dropout)
        )

        self.den_head = nn.Sequential(
            ConvBlock(self.mem_dim, 1, kernel_size=1, padding=0)
        )

    def forward_mem(self, y):
        b, k, h, w = y.shape
        m = self.mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.view(b, k, -1)
        logits = torch.bmm(m_key, y_) / sqrt(k)
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.view(b, k, h, w)

        return y_new_, logits

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)
        d = self.den_head(y_den_new)

        d = upsample(d, scale_factor=4)

        return d

class DGModel_memadd(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, err_thrs=0.5):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)

        self.err_thrs = err_thrs

        self.den_dec = nn.Sequential(
            ConvBlock(256 + 128 + 64, 256, kernel_size=1, padding=0, bn=True)
        )

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2):
        y_cat1, _ = self.forward_fe(img1)
        y_cat2, _ = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()

        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)

        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)

        d1 = upsample(d1, scale_factor=4)
        d2 = upsample(d2, scale_factor=4)

        return d1, d2, loss_con

class DGModel_cls(DGModel_base):
    def __init__(self, pretrained=True, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.cls_head = nn.Sequential(
            ConvBlock(960, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)

        return dc, c

class DGModel_memcls(DGModel_mem):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.cls_head = nn.Sequential(
            ConvBlock(960, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den_new)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)

        return dc, c

class DGModel_final(DGModel_memcls):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, cls_thrs=0.5, err_thrs=0.5, den_dropout=0.5, cls_dropout=0.3, has_err_loss=False):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, cls_dropout, cls_thrs)

        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss

        self.den_dec = nn.Sequential(
            ConvBlock(256 + 128 + 64, self.mem_dim, kernel_size=1, padding=0, bn=True)  # Corrected channels
        )

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2, c_gt=None):
        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < self.err_thrs).clone().detach()
        loss_err = F.l1_loss(y_in1, y_in2) if self.has_err_loss else 0

        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)

        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)

        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)

        c_resized_gt = self.transform_cls_map_gt(c_gt)
        c_resized1 = self.transform_cls_map_pred(c1)
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(c_resized1 - c_resized2)
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1)

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)
        dc1 = upsample(d1 * c_resized, scale_factor=4)
        dc2 = upsample(d2 * c_resized, scale_factor=4)
        c_err = upsample(c_err, scale_factor=4)

        return dc1, dc2, c1, c2, c_err, loss_con, loss_err