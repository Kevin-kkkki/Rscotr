# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead

from quant.lsq_plus import LinearLSQ , ActLSQ
from quant.Quant import Qmodes


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # self.fc = nn.Linear(self.in_channels, self.num_classes)
        # 1. 替换线性层为量化版本（__init__中修改）
        self.fc = LinearLSQ(
            in_features=self.in_channels,
            out_features=self.num_classes,
            bias=True,  # 与原nn.Linear一致（默认有bias）
            nbits_w=4,  # 权重量化位宽
            mode=Qmodes.layer_wise  # 量化模式，与其他量化层保持一致
        )

        # 2. 新增激活量化层（__init__中添加）
        self.act_quant = ActLSQ(
            in_features=self.in_channels,
            nbits_a=4  # 激活量化位宽，与权重保持一致
        )


    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.act_quant(x)
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x,task=0)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x,task=0)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

