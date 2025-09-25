import torch
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
from torch.nn.parameter import Parameter
from quant._quan_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ


__all__ = ['Conv2dQ', 'LinearQ', 'ActQ']


class FunQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
        super(Conv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
       
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = ActQ(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)

        return F.linear(x, w_q, self.bias)


class ActQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)
        # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x



class Q_attention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Q_attention, self).__init__()
        self.out_proj = LinearQ(embed_dim, embed_dim, bias=bias, nbits_w=4, **factory_kwargs)