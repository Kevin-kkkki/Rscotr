# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from quant.lsq_plus import Conv2dLSQ
from quant.Quant import Qmodes

from ..builder import NECKS


@NECKS.register_module()
class ChannelMapper(BaseModule):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None,
                 nbits_w=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ChannelMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        self.nbits_w = nbits_w
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                # ConvModule(
                #     in_channel,
                #     out_channels,
                #     kernel_size,
                #     padding=(kernel_size - 1) // 2,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg)

                Conv2dLSQ(
                in_channels=in_channel,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # 原通道调整卷积默认步长1（保持尺度不变）
                padding=(kernel_size - 1) // 2,  # 对称padding，确保输入输出尺度一致
                bias=True,  # 与原ConvModule默认bias=True保持一致
                nbits_w=self.nbits_w,  # 权重量化位宽
                mode=Qmodes.kernel_wise  # 与Conv2dLSQ的量化模式保持一致
                ))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    # ConvModule(
                    #     in_channel,
                    #     out_channels,
                    #     3,
                    #     stride=2,
                    #     padding=1,
                    #     conv_cfg=conv_cfg,
                    #     norm_cfg=norm_cfg,
                    #     act_cfg=act_cfg)
                    Conv2dLSQ(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,  # 原通道调整卷积默认步长1（保持尺度不变）
                        padding=(kernel_size - 1) // 2,  # 对称padding，确保输入输出尺度一致
                        bias=True,  # 与原ConvModule默认bias=True保持一致
                        nbits_w=self.nbits_w,  # 权重量化位宽
                        mode=Qmodes.kernel_wise))

    def forward(self, inputs,task=None):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        # outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        # if self.extra_convs:
        #     for i in range(len(self.extra_convs)):
        #         if i == 0:
        #             outs.append(self.extra_convs[0](inputs[-1]))
        #         else:
        #             outs.append(self.extra_convs[i](outs[-1]))
        # return tuple(outs)

        # -------------------------- 1. 多尺度输入通道调整（量化版） --------------------------
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]  # 当前尺度输入特征（B, C_in, H, W）
            # 调用Conv2dLSQ的forward：内置“激活量化→量化卷积”逻辑
            quant_out = self.convs[i](x, task=task)
            outs.append(quant_out)

        # -------------------------- 2. 生成额外尺度特征（量化版） --------------------------
        if self.extra_convs is not None:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    # 第一个额外层：输入为最后一个原始输入特征
                    x = inputs[-1]
                else:
                    # 后续额外层：输入为前一个量化卷积的输出（复用已有特征）
                    x = outs[-1]
                # 调用额外量化卷积层
                extra_quant_out = self.extra_convs[i](x, task=task)
                outs.append(extra_quant_out)

        return tuple(outs)
