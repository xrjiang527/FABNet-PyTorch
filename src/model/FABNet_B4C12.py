import torch.nn as nn
from .binary_common_nv1 import *
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import torch


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LL_residual_blocks(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(LL_residual_blocks, self).__init__()
        self.conv1 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)       
        self.conv2 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)

        self.move0 = LearnableBias(in_channels)
        self.move1 = LearnableBias(in_channels)

        self.relu11 = nn.PReLU(out_channels)
        self.relu12 = nn.PReLU(out_channels)
        self.aq = OursBinActivation()
        
    def forward(self, x, epoch):
        if epoch <= 50:
            itv1 = 1
            itv2 = 1
        elif epoch <= 200:
            itv1 = 2
            itv2 = 2
        else:
            itv1 = 4
            itv2 = 4
        res = x
        binary_bias1 = self.aq(self.move0(x), itv1, itv2 )
        out1 = self.conv1(binary_bias1)
        out1 = self.relu11(out1)

        binary_bias2 = self.aq(self.move1(out1), itv1, itv2)
        out2 = self.conv2(binary_bias2)
        out2 = self.relu12(out2)

        return out2 + res


class HH_residual_blocks(nn.Module):
    def __init__(self, in_channels=12*3, out_channels=12*3):
        super(HH_residual_blocks, self).__init__()
        self.conv1 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)       
        self.conv2 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)

        self.move0 = LearnableBias(in_channels)
        self.move1 = LearnableBias(in_channels)

        self.relu11 = nn.PReLU(out_channels)
        self.relu22 = nn.PReLU(out_channels)

        self.aq = OursBinActivation()

    def forward(self, x, epoch):
        if epoch <= 200:
            itv1 = 2
            itv2 = 2
        else:
            itv1 = 4
            itv2 = 4
        res = x
        binary_bias1 = self.aq(self.move0(x), itv1, itv2)
    
        out1 = self.conv1(binary_bias1)
        out1 = self.relu11(out1)

        binary_bias2 = self.aq(self.move1(out1), itv1, itv2)
        out2 = self.conv2(binary_bias2)
        out2 = self.relu22(out2)
        return out2 + res



class FABNet(nn.Module):
    def __init__(self, args):
        super(FABNet, self).__init__()

        self.n_feats = args.n_feats

        self.colors = 3

        self.resblocks = 4

        self.kernel_size = 3

        self.stride = 1

        self.padding = 1

        self.dialitation = 1

        self.conv_head = nn.Conv2d(3, self.n_feats, 3, 1, 1)

        self.scale_factor = args.scale
        

        self.DWTForward = DWTForward(wave='db1', mode='symmetric')
        self.DWTInverse = DWTInverse(wave='db1', mode='symmetric')
        
        self.body_LL_1 = LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats)
        self.body_LL_2 = LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats)
        self.body_LL_3 = LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats)
        self.body_LL_4 = LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats)


        HH_modules_body = [
        HH_residual_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3) \
            for _ in range(8)]      
        self.body_HH = nn.Sequential(*HH_modules_body)

       
        self.conv_last_h = nn.Conv2d(self.n_feats, 3, 3, 1, 1)
        self.relu = nn.PReLU()
        self.pixel = nn.PixelShuffle(2)



        self.parameters1 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters2 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters3 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters4 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters5 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        
        self.parameters1.data.fill_(1 / 5)
        self.parameters2.data.fill_(1 / 5)
        self.parameters3.data.fill_(1 / 5)
        self.parameters4.data.fill_(1 / 5)
        self.parameters5.data.fill_(1 / 5)


    def forward(self, input, epoch):
        
        input = F.interpolate(input, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        res = input
        input = self.conv_head(input)
                       
        input_wavelet_LL, input_wavelet_H  = self.DWTForward(input)
            
        b, c, n, h, w = input_wavelet_H[0].shape
        input_wavelet_H = input_wavelet_H[0].view(b, -1, h, w)
        
        res_wavelet_H = input_wavelet_H
        
        input_wavelet_LL_1 = self.body_LL_1(input_wavelet_LL,  epoch)
        input_wavelet_LL_2 = self.body_LL_2(input_wavelet_LL_1, epoch)
        
        input_wavelet_LL_3 = self.body_LL_3(input_wavelet_LL_2, epoch)
        input_wavelet_LL_4 = self.body_LL_4(input_wavelet_LL_3, epoch)
        
        input_wavelet_LL_final = self.parameters1 * input_wavelet_LL + self.parameters2 * input_wavelet_LL_1 + self.parameters3 * input_wavelet_LL_2 + self.parameters4 * input_wavelet_LL_3 + self.parameters5 * input_wavelet_LL_4 
        out_LL = input_wavelet_LL_final

        for i in range(8):
            input_wavelet_H = self.body_HH[i](input_wavelet_H, epoch)
               
        out_H = input_wavelet_H + res_wavelet_H
        out_H = out_H.view(b,c,n,h,w)
        out = self.DWTInverse((out_LL,[out_H]))
        b, c, h, w = res.shape
    
        out = self.conv_last_h(out)[:,:,:h,:w] + res
        return out      
