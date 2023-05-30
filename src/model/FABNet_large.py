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


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=1, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2 * k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

class Attention(DyReLU):
    def __init__(self, channels, reduction=48, k=1, conv_type='2d'):
        super(Attention, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

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

class HH_residual_attention_blocks(nn.Module):
    def __init__(self, in_channels=96*3, out_channels=96*3):
        super(HH_residual_attention_blocks, self).__init__()
        self.conv1 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)       
        self.conv2 = conv_ours_ds(in_channels, in_channels, 3, 1, 1)

        self.move0 = LearnableBias(in_channels)
        self.move1 = LearnableBias(in_channels)

        self.relu11 = nn.PReLU(out_channels)
        self.aq = OursBinActivation()

        self.attention = Attention(in_channels)
        
    def forward(self, x, epoch):
        if epoch <= 150:
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
        out2 = self.attention(out2)
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
        
        LL_modules_body = [
            LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats) \
            for _ in range(8)]   
        self.body_LL = nn.Sequential(*LL_modules_body)


        HH_modules_body1 = [
        HH_residual_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3) \
            for _ in range(3)]  
        HH_modules_body1.append(HH_residual_attention_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3))    
        self.body_HH1 = nn.Sequential(*HH_modules_body1)

        HH_modules_body2 = [
        HH_residual_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3) \
            for _ in range(3)]  
        HH_modules_body2.append(HH_residual_attention_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3))    
        self.body_HH2 = nn.Sequential(*HH_modules_body2)

        HH_modules_body3 = [
        HH_residual_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3) \
            for _ in range(3)]  
        HH_modules_body3.append(HH_residual_attention_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3))    
        self.body_HH3 = nn.Sequential(*HH_modules_body3)

        HH_modules_body4 = [
        HH_residual_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3) \
            for _ in range(3)]  
        HH_modules_body4.append(HH_residual_attention_blocks(in_channels=self.n_feats * 3, out_channels= self.n_feats * 3))    
        self.body_HH4 = nn.Sequential(*HH_modules_body4)

       
        self.conv_last_h = nn.Conv2d(self.n_feats, 3, 3, 1, 1)

        self.parameters1 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters2 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters3 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters4 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters5 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters6 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters7 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters8 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)
        self.parameters9 = torch.nn.Parameter(torch.FloatTensor(1, 1, 1, 1), requires_grad=True)

        self.parameters1.data.fill_(1 / 9)
        self.parameters2.data.fill_(1 / 9)
        self.parameters3.data.fill_(1 / 9)
        self.parameters4.data.fill_(1 / 9)
        self.parameters5.data.fill_(1 / 9)
        self.parameters6.data.fill_(1 / 9)
        self.parameters7.data.fill_(1 / 9)
        self.parameters8.data.fill_(1 / 9)
        self.parameters9.data.fill_(1 / 9)


    def forward(self, input, epoch):
        
        input = F.interpolate(input, scale_factor=self.scale_factor)
        
        res = input
        input = self.conv_head(input)
                       
        input_wavelet_LL, input_wavelet_H  = self.DWTForward(input)
            
        b, c, n, h, w = input_wavelet_H[0].shape
        input_wavelet_H = input_wavelet_H[0].view(b, -1, h, w)
        
        res_wavelet_H = input_wavelet_H
        
        output_wavelet_LL = []
        output_wavelet = input_wavelet_LL
        for i in range(8):
            output_wavelet = self.body_LL[i](output_wavelet, epoch)
            output_wavelet_LL.append(output_wavelet)
        input_wavelet_LL_final = self.parameters1 * input_wavelet_LL + self.parameters2 * output_wavelet_LL[0] + self.parameters3* output_wavelet_LL[1] + self.parameters4* output_wavelet_LL[2] + self.parameters5 * output_wavelet_LL[3] + self.parameters6 * output_wavelet_LL[4] + self.parameters7 * output_wavelet_LL[5] + self.parameters8 * output_wavelet_LL[6] + self.parameters9 * output_wavelet_LL[7]
        out_LL = input_wavelet_LL_final
               
        for i in range(4):
            input_wavelet_H = self.body_HH1[i](input_wavelet_H, epoch)
        for i in range(4):
            input_wavelet_H = self.body_HH2[i](input_wavelet_H, epoch)
        for i in range(4):
            input_wavelet_H = self.body_HH3[i](input_wavelet_H, epoch)
        for i in range(4):
            input_wavelet_H = self.body_HH4[i](input_wavelet_H, epoch)
        out_H = input_wavelet_H + res_wavelet_H
        out_H = out_H.view(b,c,n,h,w)
        out = self.DWTInverse((out_LL,[out_H]))

        b, c, h, w = res.shape
    
        out = self.conv_last_h(out)[:,:,:h,:w] + res
        
        
        
       
        return out      