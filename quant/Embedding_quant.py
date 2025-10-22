import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.lsq_plus import ActLSQ, round_pass, grad_scale
from quant._quan_base_plus import Qmodes


class EmbeddingLSQ(nn.Embedding):
    """
    量化版 Embedding 层（基于仓库 quant 中的 LSQ 方法）
    - 权重量化：使用 LSQ 量化，支持 kernel-wise / layer-wise
    - 激活量化：复用 quant.lsq_plus.ActLSQ
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.0,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 nbits_w=8,      # 权重量化位宽
                 nbits_a=8,      # 激活量化位宽
                 mode=Qmodes.layer_wise,  # 量化模式
                 **kwargs):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight
        )

        # 量化配置
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.mode = mode

        # 权重量化参数
        if self.nbits_w > 0:
            if self.mode == Qmodes.kernel_wise:
                # 每个 embedding 向量单独量化
                self.alpha = nn.Parameter(torch.Tensor(num_embeddings))
            else:
                # 整个权重矩阵共享一个量化参数
                self.alpha = nn.Parameter(torch.Tensor(1))
            self.register_buffer('init_state', torch.zeros(1))

        # 激活量化器
        if self.nbits_a > 0:
            self.act_quant = ActLSQ(
                in_features=embedding_dim,
                nbits_a=nbits_a,
                mode=Qmodes.layer_wise
            )
        else:
            self.act_quant = None

    def _quantize_weight(self):
        """使用权重量化（复用 LSQ 方法）"""
        if self.nbits_w <= 0:
            return self.weight

        # 量化范围
        Qn = torch.tensor(-(2 ** (self.nbits_w - 1)), dtype=torch.float32, device=self.weight.device)
        Qp = torch.tensor(2 ** (self.nbits_w - 1) - 1, dtype=torch.float32, device=self.weight.device)

        # 初始化缩放因子 alpha
        if self.training and self.init_state[0] == 0:
            if self.mode == Qmodes.kernel_wise:
                # kernel-wise：每个 embedding 向量单独初始化
                self.alpha.data.copy_(
                    2 * self.weight.abs().mean(dim=1) / torch.sqrt(Qp)
                )
            else:
                # layer-wise：整体初始化
                self.alpha.data.copy_(
                    2 * self.weight.abs().mean() / torch.sqrt(Qp)
                )
            self.init_state.fill_(1)

        # 梯度缩放
        g = 1.0 / torch.sqrt(torch.tensor(self.weight.numel(), dtype=torch.float32, device=self.weight.device) * Qp)
        alpha = grad_scale(self.alpha, g)

        # 执行量化
        if self.mode == Qmodes.kernel_wise:
            alpha = alpha.unsqueeze(1)  # [num_embeddings, 1]

        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return w_q

    def forward(self, x: torch.Tensor, task=None):
        # 1. 量化权重
        quant_weight = self._quantize_weight()

        # 2. 调用原生 embedding 查找
        out = F.embedding(
            x, quant_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

        # 3. 激活量化（如果开启）
        if self.act_quant is not None:
            out = self.act_quant(out, task=task)

        return out

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f', nbits_w={self.nbits_w}, nbits_a={self.nbits_a}, mode={self.mode}'
        return s