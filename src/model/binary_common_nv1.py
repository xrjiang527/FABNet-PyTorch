from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class conv_ours_ds(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super(conv_ours_ds, self).__init__()
    self.stride = stride
    self.padding = padding
    self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
    self.shape = (out_channels, in_channels, kernel_size, kernel_size)
    # self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
    self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

  def forward(self, x):
    real_weights = self.weight
    scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                dim=1, keepdim=True)
    
    scaling_factor = scaling_factor.detach()
    binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
    cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
    binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
    
    y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

    return y




def uniform_quantize():
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

      out = torch.sign(input)

      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class OursBinActivation(nn.Module):
    def __init__(self):
        super(OursBinActivation, self).__init__()

    def forward(self, x, itv1, itv2):

        out_forward = torch.sign(x)

        mask1 = x < -1 * (2/itv1)
        mask2 = x < 0
        mask3 = x < (2/itv2)

        out1 = (-1) * mask1.type(torch.float32) + ( 0.25 * (itv1 * itv1) * x*x + itv1 * x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-0.25 * (itv2 * itv2) * x*x + itv2 * x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))

        out = out_forward.detach() - out3.detach() + out3

        return out


