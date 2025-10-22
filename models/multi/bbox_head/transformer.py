# 版权声明：该代码版权归OpenMMLab所有，保留所有权利
# TODO：该代码基于mmcv（OpenMMLab的计算机视觉基础库）修改而来
import math  # 导入数学计算模块，用于后续三角函数等计算

import torch  # 导入PyTorch深度学习框架核心库
import torch.nn as nn  # 导入PyTorch的神经网络模块，用于构建模型层

# 从mmcv的Transformer层序列注册器中导入TRANSFORMER_LAYER_SEQUENCE，用于注册自定义Transformer层序列
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

# 从mmdet（目标检测工具箱）的Transformer工具模块中导入相关组件：
# - DeformableDetrTransformerDecoder：可变形DETR的Transformer解码器基础类
# - DeformableDetrTransformer：可变形DETR的Transformer基础类
# - inverse_sigmoid：反sigmoid函数，用于坐标变换
# - Transformer：基础Transformer类
# - build_transformer_layer_sequence：构建Transformer层序列的工具函数
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import (DeformableDetrTransformerDecoder,
                                            DeformableDetrTransformer,
                                            inverse_sigmoid, Transformer,
                                            build_transformer_layer_sequence)
from quant.lsq_plus import LinearLSQ , ActLSQ
from quant.Quant import Qmodes


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    """
    构建多层感知机（MLP）的工具函数
    
    参数:
        input_dim (int): MLP的输入特征维度
        hidden_dim (int): MLP隐藏层的特征维度
        output_dim (int): MLP的输出特征维度
        num_layers (int): MLP的总层数（需大于1）
    
    返回:
        nn.Sequential: 构建好的MLP网络（包含Linear层和ReLU激活函数）
    """
    # 断言层数必须大于1，否则抛出错误（MLP至少需要输入层+输出层，通常含隐藏层）
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    # 构建隐藏层维度列表：除输入层和输出层外，中间均为hidden_dim
    h = [hidden_dim] * (num_layers - 1)
    layers = list()  # 用于存储MLP的所有层
    # 循环构建隐藏层：输入维度从[input_dim] + h[:-1]依次取，输出维度从h依次取
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))  # 每个隐藏层由Linear+ReLU组成
    # 注意：原始DETR仓库中MLP的ReLU是inplace=False，而mmdet中FFN的ReLU默认inplace=True
    layers.append(nn.Linear(hidden_dim, output_dim))  # 添加输出层（无激活函数）
    return nn.Sequential(*layers)  # 将层列表封装为Sequential模块

def build_quant_MLP(input_dim, hidden_dim, output_dim, num_layers, 
                    nbits=4, mode=Qmodes.layer_wise):
    """构建量化版本的MLP，支持task参数传递给LinearLSQ层"""
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    
    h = [hidden_dim] * (num_layers - 1)
    layers = []
    
    # 构建隐藏层（量化线性层 + ReLU）
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend([
            LinearLSQ(
                in_features=n,
                out_features=k,
                nbits=nbits,
                mode=mode  # 量化模式（如逐层量化）
            ),
            nn.ReLU(inplace=False)  # 保持原注释中的inplace=False配置
        ])
    
    # 构建输出层（最后一层量化线性层，无激活）
    layers.append(
        LinearLSQ(
            in_features=hidden_dim,
            out_features=output_dim,
            nbits=nbits,
            mode=mode
        )
    )
    
    # 使用包装器支持task参数传递
    return QuantMLPWrapper(nn.Sequential(*layers))


class QuantMLPWrapper(nn.Module):
    """包装器类，用于在前向传播时将task参数传递给所有LinearLSQ层"""
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential  # 原始的层序列（含LinearLSQ和ReLU）
    
    def forward(self, x, task=None):
        """重写forward，仅向量化层传递task参数"""
        for layer in self.sequential:
            if isinstance(layer, LinearLSQ):
                # 量化线性层需要传入task参数
                x = layer(x, task=task)
            else:
                # ReLU等普通层直接前向传播
                x = layer(x)
        return x




@TRANSFORMER_LAYER_SEQUENCE.register_module()  # 注册为Transformer层序列，允许通过配置文件调用
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
    """
    DINO（DETR with Improved DeNoising Anchor Boxes）的Transformer解码器类
    继承自可变形DETR的解码器，主要新增参考点特征头和位置编码相关功能
    """

    def __init__(self, *args, **kwargs):
        """
        构造函数：调用父类构造函数并初始化自定义层
        参数*args和**kwargs用于传递父类所需的所有参数（如层数、嵌入维度等）
        """
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()  # 初始化自定义层（参考点特征头和归一化层）

    def _init_layers(self):
    # 原代码：self.ref_point_head = build_MLP(...)
    # 替换为量化版本，指定量化参数（如nbits=4，可根据需求调整）
        self.ref_point_head = build_quant_MLP(
            input_dim=self.embed_dims * 2,
            hidden_dim=self.embed_dims,
            output_dim=self.embed_dims,
            num_layers=2,
            nbits=4,  # 量化位数（如4位、8位）
            mode=Qmodes.layer_wise  # 量化模式（根据LinearLSQ支持的模式选择）
        )
        self.norm = nn.LayerNorm(self.embed_dims)
    # def _init_layers(self):
    #     """初始化DINO解码器特有的层结构"""
    #     # 参考点特征头：MLP网络，输入为2倍嵌入维度（位置编码输出），输出为嵌入维度，2层结构
    #     self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
    #                                     self.embed_dims, 2)
    #     # 归一化层：对解码器输出进行层归一化，稳定训练
    #     self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod  # 静态方法：无需实例化即可调用，无self参数
    def gen_sineembed_for_position(pos_tensor):
        """
        生成位置的正弦位置编码（Sinusoidal Position Embedding）
        支持2维（x,y）或4维（x,y,w,h）坐标的位置编码，是DETR系列的标准位置编码方式
        
        参数:
            pos_tensor (torch.Tensor): 位置坐标张量，形状为[bs, num_queries, 2/4]
                bs: 批次大小，num_queries: 查询框数量，2/4: 坐标维度（xy或xywh）
        
        返回:
            torch.Tensor: 正弦位置编码，形状为[bs, num_queries, 256]（2维坐标）或[bs, num_queries, 512]（4维坐标）
        """
        scale = 2 * math.pi  # 角度缩放因子，将坐标映射到[0, 2π]范围
        # 生成频率维度：128个维度，用于构建不同频率的正弦/余弦函数
        dim_t = torch.arange(
            128, dtype=torch.float32, device=pos_tensor.device)
        # 频率计算公式：10000^(-2*(dim_t//2)/128)，实现频率随维度增加指数衰减
        dim_t = 10000**(2 * (dim_t // 2) / 128)
        
        # 处理x坐标的位置编码
        x_embed = pos_tensor[:, :, 0] * scale  # x坐标 * 2π，缩放角度
        pos_x = x_embed[:, :, None] / dim_t  # [bs, num_queries, 128]：每个x坐标对应128个频率
        # 对pos_x进行正弦/余弦交替处理：偶数索引用sin，奇数索引用cos，然后展平
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)  # 展平后维度：[bs, num_queries, 128]
        
        # 处理y坐标的位置编码（与x同理）
        y_embed = pos_tensor[:, :, 1] * scale
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)  # 维度：[bs, num_queries, 128]
        
        # 根据输入坐标维度，拼接位置编码
        if pos_tensor.size(-1) == 2:
            # 2维坐标（xy）：拼接y和x的编码，总维度128+128=256
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            # 4维坐标（xywh）：额外处理w和h的编码
            # 处理w坐标
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)  # [bs, num_queries, 128]
            
            # 处理h坐标
            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)  # [bs, num_queries, 128]
            
            # 拼接y、x、w、h的编码，总维度128*4=512
            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            # 不支持的坐标维度，抛出错误
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                task=None,
                **kwargs):
        """
        DINO解码器的前向传播函数：处理查询框（query），通过多层解码层输出最终结果
        
        参数:
            query (torch.Tensor): 解码器输入查询张量，形状为[num_queries, bs, embed_dims]
            *args: 传递给解码层的其他位置参数（如key、value等）
            reference_points (torch.Tensor): 参考点坐标张量，形状为[bs, num_queries, 4]（sigmoid归一化后）
            valid_ratios (torch.Tensor): 有效区域比例张量，形状为[bs, num_levels, 2]，用于校正参考点
            reg_branches (list[nn.Module]): 回归分支列表，每个解码层对应一个回归分支，用于预测坐标偏移
            **kwargs: 传递给解码层的其他关键字参数（如注意力掩码等）
        
        返回:
            tuple: 包含以下两种情况的输出：
                1. 若return_intermediate=True（返回中间层结果）：
                   (torch.Tensor, torch.Tensor): 中间层输出序列（[num_layers, num_queries, bs, embed_dims]）、
                                               中间层参考点序列（[num_layers+1, bs, num_queries, 4]）
                2. 若return_intermediate=False：
                   (torch.Tensor, torch.Tensor): 最终层输出（[num_queries, bs, embed_dims]）、
                                               最终参考点（[bs, num_queries, 4]）
        """
        output = query  # 初始化输出为输入查询
        intermediate = []  # 存储中间层输出（若需返回）
        # 存储中间层参考点（初始为输入参考点）
        intermediate_reference_points = [reference_points]
        
        # 遍历每个解码层（lid：层索引，layer：解码层模块）
        for lid, layer in enumerate(self.layers):
            # 根据参考点维度，调整参考点（结合有效区域比例valid_ratios）
            if reference_points.shape[-1] == 4:
                # 4维参考点（xywh）：valid_ratios需重复两次（xy和wh分别校正）
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                # 2维参考点（xy）：直接与valid_ratios相乘校正
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            
            # 1. 生成参考点的正弦位置编码
            # reference_points_input[:, :, 0, :]：取第0个维度（消除level维度），形状[bs, num_queries, 4]
            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])
            
            # 2. 通过参考点特征头（MLP）处理位置编码，得到查询框的位置特征
            #query_pos = self.ref_point_head(query_sine_embed)      
            query_pos = self.ref_point_head(query_sine_embed, task=task)
            
            # 3. 调整query_pos的维度顺序：从[bs, num_queries, embed_dims]转为[num_queries, bs, embed_dims]
            # 以匹配Transformer层的输入格式（query维度顺序）
            query_pos = query_pos.permute(1, 0, 2)
            
            # 4. 调用当前解码层进行前向传播
            output = layer(
                output,  # 输入查询（上一层输出）
                *args,   # 其他位置参数（如key=memory, value=memory）
                query_pos=query_pos,  # 查询框的位置特征
                reference_points=reference_points_input,  # 校正后的参考点（用于可变形注意力）
                task=task,
                **kwargs)  # 其他关键字参数（如注意力掩码）
            
            # 5. 调整输出维度顺序：从[num_queries, bs, embed_dims]转为[bs, num_queries, embed_dims]
            # 以匹配回归分支的输入格式
            output = output.permute(1, 0, 2)
            
            # 6. 若存在回归分支，更新参考点（预测坐标偏移并应用反sigmoid变换）
            if reg_branches is not None:
                # 调用当前层的回归分支，预测坐标偏移（未经过sigmoid）
                tmp = reg_branches[lid](output)
                # 断言参考点为4维（DINO使用xywh格式）
                assert reference_points.shape[-1] == 4
                # TODO: 该步骤应更早执行（代码待优化标记）
                # 计算新参考点：偏移量 + 反sigmoid(原始参考点)，再经过sigmoid归一化
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)  # inverse_sigmoid：将sigmoid输出转回原始空间
                new_reference_points = new_reference_points.sigmoid()  # 归一化到[0,1]
                # 分离参考点（不参与梯度回传，仅作为下一层的输入）
                reference_points = new_reference_points.detach()
            
            # 7. 恢复输出维度顺序：从[bs, num_queries, embed_dims]转回[num_queries, bs, embed_dims]
            output = output.permute(1, 0, 2)
            
            # 8. 若需返回中间层结果，将当前层输出（经归一化）和参考点加入列表
            if self.return_intermediate:
                intermediate.append(self.norm(output))  # 中间层输出经层归一化
                intermediate_reference_points.append(new_reference_points)
                # NOTE: 这是为了"Look Forward Twice"模块（DINO的改进之一），在DeformDETR中仅参考点会被记录
        
        # 根据是否返回中间层结果，返回对应输出
        if self.return_intermediate:
            # 中间层输出：堆叠为[num_layers, num_queries, bs, embed_dims]
            # 中间层参考点：堆叠为[num_layers+1, bs, num_queries, 4]
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        
        # 不返回中间层：返回最终层输出和最终参考点
        return output, reference_points


@TRANSFORMER.register_module()  # 注册为Transformer模块，允许通过配置文件调用
class DinoTransformer(DeformableDetrTransformer):
    """
    DINO的完整Transformer类（包含编码器和解码器）
    继承自可变形DETR的Transformer，主要适配DINO的两阶段检测流程和去噪训练
    """

    def __init__(self,
                 decoder=None,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 init_cfg=None):
        """
        构造函数：初始化DINO Transformer的编码器、解码器及相关组件
        
        参数:
            decoder (dict): 解码器配置字典，用于构建DinoTransformerDecoder
            as_two_stage (bool): 是否使用两阶段检测流程（DINO必须为True）
            num_feature_levels (int): 特征金字塔的层级数量（默认4层，如C3-C6）
            two_stage_num_proposals (int): 两阶段中第一阶段（编码器）生成的候选框数量（默认300）
            init_cfg (dict): 初始化配置字典，用于模型参数初始化
        """
        # 注意：此处调用的是基础Transformer类的构造函数（而非DeformableDetrTransformer）
        super(Transformer, self).__init__(init_cfg=init_cfg)
        # 构建解码器：通过配置字典构建DinoTransformerDecoder
        self.decoder = build_transformer_layer_sequence(decoder)
        self.as_two_stage = as_two_stage  # 两阶段标记（DINO强制为True）
        self.num_feature_levels = num_feature_levels  # 特征层级数量
        self.two_stage_num_proposals = two_stage_num_proposals  # 编码器候选框数量
        self.embed_dims = self.decoder.embed_dims  # 嵌入维度（与解码器一致）
        self.init_layers()  # 初始化自定义层（特征层级嵌入、编码器输出处理、查询嵌入）

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        # 特征层级嵌入（Level Embedding）：每个特征层级对应一个可学习的嵌入向量
        # 用于区分不同金字塔层级的特征，形状为[num_feature_levels, embed_dims]
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        # 编码器输出线性层：将编码器输出特征映射到嵌入维度（确保与解码器输入维度一致）
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        # 编码器输出归一化层：对编码器输出进行层归一化，稳定训练
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        # 查询嵌入（Query Embedding）：两阶段中解码器的初始查询嵌入，可学习
        # 形状为[two_stage_num_proposals, embed_dims]，对应300个初始查询
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)

    def init_weights(self):
        """初始化模型权重：调用父类初始化方法，并额外初始化查询嵌入"""
        super().init_weights()
        # 初始化查询嵌入的权重：使用正态分布（均值0，方差1）
        nn.init.normal_(self.query_embed.weight.data)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                encoder,
                reg_branches=None,
                cls_branches=None,
                task=None,
                **kwargs):
        """
        DINO Transformer的完整前向传播：包含编码器处理、候选框生成、解码器处理
        
        参数:
            mlvl_feats (list[torch.Tensor]): 多尺度特征列表，每个元素形状为[bs, c, h, w]
                c: 特征通道数（与embed_dims一致），h/w: 特征图高/宽
            mlvl_masks (list[torch.Tensor]): 多尺度特征掩码列表，每个元素形状为[bs, h, w]
                掩码值为True表示无效区域（如padding区域），False表示有效区域
            query_embed (None): 原始DETR的查询嵌入（DINO中未使用，需为None）
            mlvl_pos_embeds (list[torch.Tensor]): 多尺度位置编码列表，每个元素形状为[bs, c, h, w]
            dn_label_query (torch.Tensor | None): 去噪训练的标签查询嵌入，形状为[num_dn_queries, bs, embed_dims]
                用于DINO的去噪训练策略，None表示不使用去噪
            dn_bbox_query (torch.Tensor | None): 去噪训练的边界框查询，形状为[bs, num_dn_queries, 4]
                与dn_label_query对应，用于去噪训练的参考点初始化
            attn_mask (torch.Tensor | None): 注意力掩码，用于解码器的注意力层，形状根据具体实现而定
            encoder (nn.Module): Transformer编码器模块（如可变形注意力编码器）
            reg_branches (list[nn.Module]): 回归分支列表，每个解码器层对应一个，用于预测边界框偏移
            cls_branches (list[nn.Module]): 分类分支列表，每个解码器层对应一个，用于预测类别概率
            **kwargs: 传递给编码器/解码器的其他关键字参数
        
        返回:
            tuple: 包含以下4个元素的元组：
                1. inter_states (torch.Tensor): 解码器中间层输出，形状为[num_layers, num_queries, bs, embed_dims]
                2. inter_references_out (torch.Tensor): 解码器中间层参考点，形状为[num_layers+1, bs, num_queries, 4]
                3. topk_score (torch.Tensor): 编码器生成的候选框得分，形状为[bs, two_stage_num_proposals, num_classes]
                4. topk_anchor (torch.Tensor): 编码器生成的候选框坐标（sigmoid归一化后），形状为[bs, two_stage_num_proposals, 4]
        """
        # 断言：DINO必须使用两阶段流程，且query_embed必须为None（DINO用自身的query_embed）
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        # -------------------------- 步骤1：处理多尺度特征，准备编码器输入 --------------------------
        feat_flatten = []  # 展平后的多尺度特征列表
        mask_flatten = []  # 展平后的多尺度掩码列表
        lvl_pos_embed_flatten = []  # 展平后的多尺度层级位置编码列表
        spatial_shapes = []  # 各尺度特征图的空间形状列表（h, w）
        
        # 遍历每个特征层级（lvl：层级索引，feat：特征，mask：掩码，pos_embed：位置编码）
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape  # 特征形状：[bs, c, h, w]
            spatial_shape = (h, w)  # 当前层级的空间形状
            spatial_shapes.append(spatial_shape)
            
            # 特征展平：从[bs, c, h, w] → [bs, h*w, c]（展平h和w维度）
            feat = feat.flatten(2).transpose(1, 2)
            # 掩码展平：从[bs, h, w] → [bs, h*w]
            mask = mask.flatten(1)
            # 位置编码展平：从[bs, c, h, w] → [bs, h*w, c]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) 
            
            # 层级位置编码：将当前层级的位置编码与层级嵌入相加（区分不同层级）
            # self.level_embeds[lvl]：[embed_dims] → 扩展为[1, 1, embed_dims]，与pos_embed广播相加
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            
            # 将处理后的特征、掩码、层级位置编码加入列表
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        
        # 拼接所有层级的展平特征：[bs, sum(h*w across levels), c]
        feat_flatten = torch.cat(feat_flatten, 1)
        # 拼接所有层级的展平掩码：[bs, sum(h*w across levels)]
        mask_flatten = torch.cat(mask_flatten, 1)
        # 拼接所有层级的展平位置编码：[bs, sum(h*w across levels), c]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # 将空间形状列表转为张量：[num_levels, 2]（2表示h和w），用于编码器可变形注意力
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        # 计算每个层级的起始索引：用于区分不同层级的特征在展平后的位置
        # 例如：若层级1有h1*w1个像素，层级2有h2*w2个像素，则起始索引为[0, h1*w1]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算每个层级的有效区域比例：[bs, num_levels, 2]，用于校正参考点
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # -------------------------- 步骤2：生成编码器参考点 --------------------------
        # 生成可变形注意力的参考点：[bs, sum(h*w across levels), num_levels, 2]
        # 参考点坐标为特征图的归一化坐标（[0,1]范围）
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        # -------------------------- 步骤3：编码器前向传播 --------------------------
        # 调整特征和位置编码的维度顺序：从[bs, num_pixels, c] → [num_pixels, bs, c]
        # 以匹配Transformer编码器的输入格式（query维度顺序）
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        
        # 编码器前向传播：处理展平特征，输出memory（用于解码器的key和value）
        memory = encoder(                   #已经量化
            query=feat_flatten,  # 编码器输入查询（即展平特征）
            key=None,  # 可变形注意力中key与query相同，故设为None
            value=None,  # 可变形注意力中value与query相同，故设为None
            query_pos=lvl_pos_embed_flatten,  # 查询的位置编码（层级位置编码）
            query_key_padding_mask=mask_flatten,  # 查询的填充掩码（无效区域）
            spatial_shapes=spatial_shapes,  # 各层级空间形状
            reference_points=reference_points,  # 参考点（用于可变形注意力采样）
            level_start_index=level_start_index,  # 各层级起始索引
            valid_ratios=valid_ratios,  # 有效区域比例
            task=task,
            **kwargs)  # 其他编码器参数

        # -------------------------- 步骤4：编码器输出处理，生成候选框 --------------------------
        # 调整memory维度顺序：从[num_pixels, bs, c] → [bs, num_pixels, c]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape  # bs：批次大小，_：总像素数，c：嵌入维度
        
        # 生成编码器输出候选框：调用父类方法，从memory中提取候选框特征和初始坐标
        output_memory, output_proposals = self.gen_encoder_output_proposals(        
            memory, mask_flatten, spatial_shapes, task)
        
        # 编码器分类分支：使用解码器最后一层的分类分支（共享权重）预测候选框类别
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        # 编码器回归分支：预测候选框坐标偏移，与初始坐标相加得到未归一化的坐标
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        
        # 提取Top-K候选框（默认300个）：根据分类得分选择置信度最高的候选框
        cls_out_features = cls_branches[self.decoder.num_layers].out_features  # 类别数量
        topk = self.two_stage_num_proposals  # 需选择的候选框数量
        # 对每个样本，取分类得分最高的topk个候选框的索引：[bs, topk]
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        # 根据索引提取topk个候选框的分类得分：[bs, topk, num_classes]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        # 根据索引提取topk个候选框的未归一化坐标：[bs, topk, 4]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        # 候选框坐标归一化（sigmoid）：[bs, topk, 4]，范围[0,1]
        topk_anchor = topk_coords_unact.sigmoid()
        # 分离未归一化坐标（不参与梯度回传，作为解码器初始参考点）
        topk_coords_unact = topk_coords_unact.detach()

        # -------------------------- 步骤5：初始化解码器查询和参考点 --------------------------
        # 初始化解码器查询：从查询嵌入（self.query_embed）扩展到批次维度
        # self.query_embed.weight：[topk, embed_dims] → 扩展为[topk, bs, embed_dims] → 转置为[bs, topk, embed_dims]
        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        
        # 若存在去噪查询（dn_label_query），拼接去噪查询和正常查询
        if dn_label_query is not None:
            # dn_label_query：[bs, num_dn_queries, embed_dims] → 拼接后[bs, num_dn_queries+topk, embed_dims]
            query = torch.cat([dn_label_query, query], dim=1)
        
        # 若存在去噪边界框查询（dn_bbox_query），拼接去噪参考点和正常参考点
        if dn_bbox_query is not None:
            # dn_bbox_query：[bs, num_dn_queries, 4] → 拼接后[bs, num_dn_queries+topk, 4]
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            # 无去噪时，参考点为编码器生成的topk候选框（未归一化）
            reference_points = topk_coords_unact
        
        # 参考点归一化（sigmoid）：转为[0,1]范围，作为解码器初始参考点
        reference_points = reference_points.sigmoid()

        # -------------------------- 步骤6：解码器前向传播 --------------------------
        # 调整query和memory的维度顺序：匹配解码器输入格式（[num_queries, bs, embed_dims]）
        query = query.permute(1, 0, 2)  # [num_queries, bs, embed_dims]
        memory = memory.permute(1, 0, 2)  # [num_pixels, bs, embed_dims]
        
        # 解码器前向传播：输出中间层状态和中间层参考点
        inter_states, inter_references = self.decoder(
            query=query,  # 解码器输入查询
            key=None,  # 解码器key为memory，故设为None
            value=memory,  # 解码器value为编码器输出memory
            attn_masks=attn_mask,  # 注意力掩码
            key_padding_mask=mask_flatten,  # key的填充掩码（无效区域）
            reference_points=reference_points,  # 初始参考点
            spatial_shapes=spatial_shapes,  # 各层级空间形状
            level_start_index=level_start_index,  # 各层级起始索引
            valid_ratios=valid_ratios,  # 有效区域比例
            reg_branches=reg_branches,  # 回归分支（用于更新参考点）
            task=task,
            **kwargs)  # 其他解码器参数

        # 解码器中间层参考点输出（直接沿用解码器返回的结果）
        inter_references_out = inter_references

        # -------------------------- 步骤7：返回最终结果 --------------------------
        # 返回解码器中间层状态、中间层参考点、编码器Top-K候选框得分、编码器Top-K候选框坐标
        return inter_states, inter_references_out, topk_score, topk_anchor




# # Copyright (c) OpenMMLab. All rights reserved.
# # TODO: Modified from mmcv
# import math

# import torch
# import torch.nn as nn

# from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

# from mmdet.models.utils.builder import TRANSFORMER
# from mmdet.models.utils.transformer import (DeformableDetrTransformerDecoder,
#                                             DeformableDetrTransformer,
#                                             inverse_sigmoid, Transformer,
#                                             build_transformer_layer_sequence)


# def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
#     assert num_layers > 1, \
#         f'num_layers should be greater than 1 but got {num_layers}'
#     h = [hidden_dim] * (num_layers - 1)
#     layers = list()
#     for n, k in zip([input_dim] + h[:-1], h):
#         layers.extend((nn.Linear(n, k), nn.ReLU()))
#     # Note that the relu func of MLP in original DETR repo is set
#     # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
#     # 'inplace=True' by default.
#     layers.append(nn.Linear(hidden_dim, output_dim))
#     return nn.Sequential(*layers)


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

#     def __init__(self, *args, **kwargs):
#         super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
#         self._init_layers()

#     def _init_layers(self):
#         self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
#                                         self.embed_dims, 2)
#         self.norm = nn.LayerNorm(self.embed_dims)

#     @staticmethod
#     def gen_sineembed_for_position(pos_tensor):
#         scale = 2 * math.pi
#         dim_t = torch.arange(
#             128, dtype=torch.float32, device=pos_tensor.device)
#         dim_t = 10000**(2 * (dim_t // 2) / 128)
#         x_embed = pos_tensor[:, :, 0] * scale
#         y_embed = pos_tensor[:, :, 1] * scale
#         pos_x = x_embed[:, :, None] / dim_t
#         pos_y = y_embed[:, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
#                             dim=3).flatten(2)
#         pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
#                             dim=3).flatten(2)
#         if pos_tensor.size(-1) == 2:
#             pos = torch.cat((pos_y, pos_x), dim=2)
#         elif pos_tensor.size(-1) == 4:
#             w_embed = pos_tensor[:, :, 2] * scale
#             pos_w = w_embed[:, :, None] / dim_t
#             pos_w = torch.stack(
#                 (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
#                 dim=3).flatten(2)

#             h_embed = pos_tensor[:, :, 3] * scale
#             pos_h = h_embed[:, :, None] / dim_t
#             pos_h = torch.stack(
#                 (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
#                 dim=3).flatten(2)

#             pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
#         else:
#             raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
#                 pos_tensor.size(-1)))
#         return pos

#     def forward(self,
#                 query,
#                 *args,
#                 reference_points=None,
#                 valid_ratios=None,
#                 reg_branches=None,
#                 **kwargs):
#         output = query
#         intermediate = []
#         intermediate_reference_points = [reference_points]
#         for lid, layer in enumerate(self.layers):
#             if reference_points.shape[-1] == 4:
#                 reference_points_input = \
#                     reference_points[:, :, None] * torch.cat(
#                         [valid_ratios, valid_ratios], -1)[:, None]
#             else:
#                 assert reference_points.shape[-1] == 2
#                 reference_points_input = \
#                     reference_points[:, :, None] * valid_ratios[:, None]

#             query_sine_embed = self.gen_sineembed_for_position(
#                 reference_points_input[:, :, 0, :])
#             query_pos = self.ref_point_head(query_sine_embed)

#             query_pos = query_pos.permute(1, 0, 2)
#             output = layer(
#                 output,
#                 *args,
#                 query_pos=query_pos,
#                 reference_points=reference_points_input,
#                 **kwargs)
#             output = output.permute(1, 0, 2)

#             if reg_branches is not None:
#                 tmp = reg_branches[lid](output)
#                 assert reference_points.shape[-1] == 4
#                 # TODO: should do earlier
#                 new_reference_points = tmp + inverse_sigmoid(
#                     reference_points, eps=1e-3)
#                 new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()

#             output = output.permute(1, 0, 2)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))
#                 intermediate_reference_points.append(new_reference_points)
#                 # NOTE this is for the "Look Forward Twice" module,
#                 # in the DeformDETR, reference_points was appended.

#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(
#                 intermediate_reference_points)

#         return output, reference_points


# @TRANSFORMER.register_module()
# class DinoTransformer(DeformableDetrTransformer):

#     def __init__(self,
#                  decoder=None,
#                  as_two_stage=False,
#                  num_feature_levels=4,
#                  two_stage_num_proposals=300,
#                  init_cfg=None):
#         super(Transformer, self).__init__(init_cfg=init_cfg)
#         self.decoder = build_transformer_layer_sequence(decoder)
#         self.as_two_stage = as_two_stage
#         self.num_feature_levels = num_feature_levels
#         self.two_stage_num_proposals = two_stage_num_proposals
#         self.embed_dims = self.decoder.embed_dims
#         self.init_layers()

#     def init_layers(self):
#         """Initialize layers of the DinoTransformer."""
#         self.level_embeds = nn.Parameter(
#             torch.Tensor(self.num_feature_levels, self.embed_dims))
#         self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
#         self.enc_output_norm = nn.LayerNorm(self.embed_dims)
#         self.query_embed = nn.Embedding(self.two_stage_num_proposals,
#                                         self.embed_dims)

#     def init_weights(self):
#         super().init_weights()
#         nn.init.normal_(self.query_embed.weight.data)

#     def forward(self,
#                 mlvl_feats,
#                 mlvl_masks,
#                 query_embed,
#                 mlvl_pos_embeds,
#                 dn_label_query,
#                 dn_bbox_query,
#                 attn_mask,
#                 encoder,
#                 reg_branches=None,
#                 cls_branches=None,
#                 **kwargs):
#         assert self.as_two_stage and query_embed is None, \
#             'as_two_stage must be True for DINO'

#         feat_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (feat, mask, pos_embed) in enumerate(
#                 zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
#             bs, c, h, w = feat.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             feat = feat.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             feat_flatten.append(feat)
#             mask_flatten.append(mask)
#         feat_flatten = torch.cat(feat_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=feat_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros(
#             (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack(
#             [self.get_valid_ratio(m) for m in mlvl_masks], 1)

#         reference_points = self.get_reference_points(
#             spatial_shapes, valid_ratios, device=feat.device)

#         feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
#         lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
#             1, 0, 2)  # (H*W, bs, embed_dims)
#         memory = encoder(
#             query=feat_flatten,
#             key=None,
#             value=None,
#             query_pos=lvl_pos_embed_flatten,
#             query_key_padding_mask=mask_flatten,
#             spatial_shapes=spatial_shapes,
#             reference_points=reference_points,
#             level_start_index=level_start_index,
#             valid_ratios=valid_ratios,
#             **kwargs)

#         memory = memory.permute(1, 0, 2)
#         bs, _, c = memory.shape

#         output_memory, output_proposals = self.gen_encoder_output_proposals(
#             memory, mask_flatten, spatial_shapes)
#         enc_outputs_class = cls_branches[self.decoder.num_layers](
#             output_memory)
#         enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
#             output_memory) + output_proposals
#         cls_out_features = cls_branches[self.decoder.num_layers].out_features
#         topk = self.two_stage_num_proposals
#         topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
#         topk_score = torch.gather(
#             enc_outputs_class, 1,
#             topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
#         topk_coords_unact = torch.gather(
#             enc_outputs_coord_unact, 1,
#             topk_indices.unsqueeze(-1).repeat(1, 1, 4))
#         topk_anchor = topk_coords_unact.sigmoid()
#         topk_coords_unact = topk_coords_unact.detach()

#         query = self.query_embed.weight[:, None, :].repeat(1, bs,
#                                                            1).transpose(0, 1)
#         if dn_label_query is not None:
#             query = torch.cat([dn_label_query, query], dim=1)
#         if dn_bbox_query is not None:
#             reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
#                                          dim=1)
#         else:
#             reference_points = topk_coords_unact
#         reference_points = reference_points.sigmoid()

#         # decoder
#         query = query.permute(1, 0, 2)
#         memory = memory.permute(1, 0, 2)
#         inter_states, inter_references = self.decoder(
#             query=query,
#             key=None,
#             value=memory,
#             attn_masks=attn_mask,
#             key_padding_mask=mask_flatten,
#             reference_points=reference_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             valid_ratios=valid_ratios,
#             reg_branches=reg_branches,
#             **kwargs)

#         inter_references_out = inter_references

#         return inter_states, inter_references_out, topk_score, topk_anchor