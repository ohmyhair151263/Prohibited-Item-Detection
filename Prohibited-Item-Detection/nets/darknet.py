#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F

# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#         print('Mish activation loaded...')
#     def forward(self,x):
#         return x*(torch.tanh(F.softplus(x)))


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    # elif name =='mish':
    #     module = Mish()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))






class DWConv(nn.Module):
    '''
    DW?????? ????????????(Depthwise) ???(Pointwise)
    DW???????????????????????????????????????????????????
    ??????Depthwise


    '''
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class DilatedConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride, groups=1, bias = False, dilation = 1, act="silu"):
        super().__init__()
        pad  = (kernel_size + (kernel_size - 1) * dilation - 1)// 2  - 1
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding= pad, bias = bias, dilation = dilation)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
class DCM(nn.Module):
    '''
    ?????????????????????????????????
    ?????????????????? 1,2,5???????????????????????????

    '''
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 5), dilated_size=(1, 2, 3), activation = 'silu' ):
        super().__init__()
        self.dcm1 = DilatedConv(in_channels,out_channels,kernel_size=3,stride = 1,dilation =1,act = activation )
        self.dcm2 = DilatedConv(in_channels, out_channels,kernel_size= 3, stride=1, dilation=3, act=activation)
        self.dcm3 = DilatedConv(in_channels, out_channels, kernel_size=3, stride=1, dilation=5,act=activation)
        print('size',kernel_size)
        conv2_channels = in_channels * (len(kernel_size) + 1)
        self.conv = BaseConv(conv2_channels,out_channels, 1, stride=1, act=activation)

    def forward(self,x) :
        # ????????????1???????????????
        # print(x.shape)

        x1 = self.dcm1(x)
        # print(x1.shape)
        # ????????????2???????????????
        x2 = self.dcm2(x)
        # print(x2.shape)
        # print(x2.shape)
        # ????????????5???????????????
        x3 = self.dcm3(x)
        # print(x3.shape)
        y = torch.cat([x, x1, x2, x3],dim = 1)
        # print('in_shape',y.shape)
        y = self.conv(y)
        # print('out_shape',y.shape)
        return  y
class SPPBottleneck(nn.Module):
    '''



    '''
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        # ??????1*1??????????????????????????????
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        # ????????????????????????
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

#--------------------------------------------------#
#   ??????????????????????????????????????????
#--------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        #--------------------------------------------------#
        #   ??????1x1???????????????????????????????????????????????????50%
        #--------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   ??????3x3?????????????????????????????????????????????????????????
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
# CSPLayer ??????????????????????????????????????????
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   ???????????????????????????
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   ????????????????????????????????????
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # print('hidden_channels,out_channels',hidden_channels,out_channels)
        #--------------------------------------------------#
        #   ?????????????????????????????????Bottleneck????????????
        #   ???????????????Sequential ??????
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1???????????????
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2????????????????????????
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   ????????????????????????????????????????????????????????????
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   ????????????????????????????????????????????????
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------#
        return self.conv3(x)

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        #-----------------------------------------------#
        #   ???????????????640, 640, 3
        #   ????????????????????????64
        #-----------------------------------------------#
        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3
        
        #-----------------------------------------------#
        #   ??????focus??????????????????????????????
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        #-----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        #-----------------------------------------------#
        #   ?????????????????????320, 320, 64 -> 160, 160, 128
        #   ??????CSPlayer?????????160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   ?????????????????????160, 160, 128 -> 80, 80, 256
        #   ??????CSPlayer?????????80, 80, 256 -> 80, 80, 256
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   ?????????????????????80, 80, 256 -> 40, 40, 512
        #   ??????CSPlayer?????????40, 40, 512 -> 40, 40, 512
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   ?????????????????????40, 40, 512 -> 20, 20, 1024
        #   ??????SPP?????????20, 20, 1024 -> 20, 20, 1024
        #   ??????CSPlayer?????????20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        #-----------------------------------------------#
        #   dark3????????????80, 80, 256???????????????????????????
        #-----------------------------------------------#
        x = self.dark3(x)
        outputs["dark3"] = x
        #-----------------------------------------------#
        #   dark4????????????40, 40, 512???????????????????????????
        #-----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x
        #-----------------------------------------------#
        #   dark5????????????20, 20, 1024???????????????????????????
        #-----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

# ????????????psa ???????????????
# ????????????SE???????????????
class SEWeightModule2(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = SiLU()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight * x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu =  nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = SiLU()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out





class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out

class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp



if __name__ == '__main__':
    print(CSPDarknet(1, 1))