# 导入PyTorch核心模块，用于构建神经网络和实现基本运算
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.Embedding_quant import EmbeddingLSQ
from quant.Quant import Qmodes
from quant.lsq_plus import ActLSQ

# 从mmdet.core导入目标检测核心工具函数：
# - bbox_cxcywh_to_xyxy/bbox_xyxy_to_cxcywh：边界框格式转换（中心宽高→对角坐标/反之）
# - multi_apply：并行将函数应用到多个输入列表，提升效率
# - reduce_mean：分布式训练中跨设备计算均值，保证数据一致性
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply,
                        reduce_mean)
# 从mmdet.models.utils.transformer导入sigmoid逆运算函数，用于边界框参考点还原
from mmdet.models.utils.transformer import inverse_sigmoid
# 从mmdet.models.builder导入HEADS注册器，用于将当前类注册为检测头模块（可通过配置调用）
from mmdet.models.builder import HEADS

# 导入父类检测头DeformableDETRHead（可变形DETR的基础检测头）和去噪查询生成器构建函数
from .mmdet_detr_head import DeformableDETRHead
from .query_denoising import build_dn_generator


# 使用HEADS注册器装饰类，使其成为可配置的检测头模块，配置中可通过type='DINOHead'调用
@HEADS.register_module()
class DINOHead(DeformableDETRHead):
    """DINO目标检测头，继承自可变形DETR检测头，核心增强去噪训练（Denoising Training）功能"""

    def __init__(self,
                 *args,
                 num_query=100,          # 解码器查询（query）的数量，默认100
                 dn_cfg=None,           # 去噪训练配置字典（如噪声强度、分组数等）
                 transformer=None,      # Transformer解码器的配置字典
                 **kwargs):
        """
        类构造函数，初始化DINO检测头的核心参数和组件
        Args:
            *args: 传递给父类DeformableDETRHead的位置参数（如分类数、特征维度等）
            num_query: 解码器查询数量，控制候选目标的数量
            dn_cfg: 去噪训练配置，控制去噪查询的生成逻辑
            transformer: Transformer解码器配置，定义解码器的层结构、注意力机制等
            **kwargs: 传递给父类DeformableDETRHead的关键字参数（如损失函数配置等）
        """
        # 适配两阶段检测：确保Transformer配置中的“两阶段提议数”与查询数一致
        if 'two_stage_num_proposals' in transformer:
            # 断言校验：两阶段提议数必须等于查询数，否则抛出错误
            assert transformer['two_stage_num_proposals'] == num_query, \
                'DINO算法要求two_stage_num_proposals必须等于num_query'
        else:
            # 若未配置两阶段提议数，默认设为查询数
            transformer['two_stage_num_proposals'] = num_query
        # 调用父类DeformableDETRHead的构造函数，初始化基础检测头组件（如分类分支、回归分支）
        super(DINOHead, self).__init__(
            *args, num_query=num_query, transformer=transformer, **kwargs)

        # DINO算法强制启用两阶段检测和边界框精修，此处断言确保配置正确
        assert self.as_two_stage, \
            'DINO算法必须启用两阶段检测（as_two_stage=True）'
        assert self.with_box_refine, \
            'DINO算法必须启用边界框精修（with_box_refine=True）'
        # 初始化自定义层（如类别嵌入层）
        self._init_layers()
        # 初始化去噪查询生成器，基于dn_cfg配置构建
        self.init_denoising(dn_cfg)

    def _init_layers(self):
        """初始化DINO特有的自定义层：类别嵌入层（label_embedding）"""
        # 先调用父类的_init_layers方法，确保父类定义的基础层（如分类/回归分支）已初始化
        super()._init_layers()
        # 注：原始DINO代码中COCO数据集的num_embeddings设为92（91个目标类+1个Unknown类），
        # 但Unknown类的嵌入向量未实际使用，此处简化为num_classes（仅目标类）
        # 类别嵌入层：将类别ID（0~num_classes-1）映射为embed_dims维的特征向量
        
        # self.label_embedding = nn.Embedding(self.num_classes,
        #                                      self.embed_dims)
        # 替换为量化版 EmbeddingLSQ
        self.label_embedding = EmbeddingLSQ(
            num_embeddings=self.num_classes,  # 类别总数（与原一致）
            embedding_dim=self.embed_dims,   # 嵌入维度（与原一致）
            nbits_w=8  # 权重量化位宽（推荐8bit，若精度足够可尝试4bit）
        )
        
    def init_denoising(self, dn_cfg):
        """基于dn_cfg配置，初始化去噪查询生成器（dn_generator）"""
        if dn_cfg is not None:
            # 为去噪配置补充必要参数：类别数、查询数、特征维度（与检测头一致）
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_queries'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        # 调用build_dn_generator构建去噪生成器，无配置时设为None
        self.dn_generator = build_dn_generator(dn_cfg)

    def forward_train(self,
                      mlvl_feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      shared_encoder=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        训练阶段前向传播函数：生成去噪查询→调用核心forward→计算损失
        Args:
            mlvl_feats: 多尺度特征图列表，来自骨干网络输出
            img_metas: 图像元信息列表，每个元素包含单张图像的尺寸、缩放比等信息
            gt_bboxes: 真实边界框列表，每个元素为单张图像的真实边界框（tensor）
            gt_labels: 真实类别标签列表，每个元素为单张图像的真实类别（tensor）
            gt_bboxes_ignore: 需忽略的真实边界框列表（如遮挡严重的目标）
            shared_encoder: 共享编码器（多任务场景下复用，单检测任务可忽略）
            proposal_cfg: 提议生成配置（DINO不依赖外部提议，故需为None）
            **kwargs: 额外参数（如训练相关的超参数）
        Returns:
            损失字典，包含各类损失项（如分类损失、边界框损失、IoU损失等）
        """
        # DINO不使用外部提议，故断言proposal_cfg必须为None
        assert proposal_cfg is None, '"proposal_cfg"必须为None（DINO无需外部提议）'
        # 训练阶段必须启用去噪，故断言dn_generator不能为None
        assert self.dn_generator is not None, '"dn_cfg"必须设置（DINO训练需去噪）'
        # 调用去噪生成器，生成带噪声的类别查询、边界框查询，及注意力掩码    已经量化
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(gt_bboxes, gt_labels,
                              self.label_embedding, img_metas)
        # 调用核心forward函数，传入去噪查询和注意力掩码，获取模型输出
        outs = self(shared_encoder, mlvl_feats, img_metas,
                    dn_label_query, dn_bbox_query, attn_mask)
        # 根据是否有真实类别标签，组装损失计算的输入参数
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, dn_meta)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, dn_meta)
        # 调用loss函数计算总损失，并返回损失字典
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, feats, img_metas, shared_encoder=None, rescale=False):
        """
        推理阶段前向传播函数：无去噪查询→调用核心forward→生成检测结果
        Args:
            feats: 多尺度特征图列表（同训练阶段）
            img_metas: 图像元信息列表（同训练阶段）
            shared_encoder: 共享编码器（同训练阶段）
            rescale: 是否将检测框坐标缩放回图像原始尺寸（推理时通常设为True）
        Returns:
            检测结果列表，每个元素为单张图像的检测结果（含类别、坐标、置信度）
        """
        # 调用核心forward函数（无去噪查询），获取模型输出
        outs = self.forward(shared_encoder, feats, img_metas)
        # 调用get_bboxes函数，将模型输出解析为最终检测框（含坐标缩放）
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def forward(self,
                encoder,
                mlvl_feats,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None):
        """
        核心前向传播函数：处理多尺度特征→生成掩码/位置编码→调用Transformer→解析输出
        Args:
            encoder: 编码器（多任务共享或独立编码器）
            mlvl_feats: 多尺度特征图列表
            img_metas: 图像元信息列表
            dn_label_query: 去噪类别查询（训练阶段有值，推理阶段为None）
            dn_bbox_query: 去噪边界框查询（训练阶段有值，推理阶段为None）
            attn_mask: 注意力掩码（用于屏蔽去噪查询的干扰，训练阶段有值）
        Returns:
            多轮解码器输出的类别分数、边界框，及编码器topk提议的分数和锚点
        """
        # 获取批次大小（多尺度特征图的第一个维度均为批次大小）
        batch_size = mlvl_feats[0].size(0)
        # 获取模型输入图像的统一尺寸（所有图像已缩放至该尺寸）
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        # 初始化图像掩码：无效区域（超出原始图像的部分）设为1，有效区域设为0
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            # 获取单张图像的原始尺寸（未缩放前）
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            # 将原始图像范围内的区域设为0（有效区域），超出部分保持1（无效区域）
            img_masks[img_id, :img_h, :img_w] = 0

        # 生成多尺度特征对应的掩码和位置编码
        mlvl_masks = []          # 多尺度特征掩码列表
        mlvl_positional_encodings = []  # 多尺度位置编码列表
        for feat in mlvl_feats:
            # 将图像掩码缩放到当前特征图的尺寸，并转为bool类型（True表示无效区域）
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            # 为当前特征图生成位置编码（基于掩码，无效区域编码为0）
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        # DINO不使用预定义查询嵌入，故设为None
        query_embeds = None
        # 调用Transformer解码器，传入多尺度特征、掩码、位置编码及去噪查询，获取输出
        hs, inter_references, topk_score, topk_anchor = \
            self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                encoder,  # 传入编码器（多任务共享或独立使用）
                # 传入边界框精修分支（启用时）
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                # 传入两阶段分类分支（启用时）
                cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                task=1
            )
        # 调整解码器输出hs的维度顺序：适配后续分类/回归分支的输入格式
        hs = hs.permute(0, 2, 1, 3)

        # 兼容无目标图像：分布式训练中，若图像无目标，类别嵌入层参数未被使用会报错，
        # 此处添加一个微小的0值操作，确保参数参与计算（不影响结果）
        if dn_label_query is not None and dn_label_query.size(1) == 0:
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        # 初始化列表，存储多轮解码器的类别分数和边界框预测
        outputs_classes = []
        outputs_coords = []

        # 遍历每轮解码器输出，生成类别分数和边界框
        for lvl in range(hs.shape[0]):
            # 获取当前轮的边界框参考点（用于精修）
            reference = inter_references[lvl]
            # 对参考点执行sigmoid逆运算：将[0,1]归一化坐标还原为原始数值范围
            reference = inverse_sigmoid(reference, eps=1e-3)
            # 调用当前轮分类分支，预测类别分数
            outputs_class = self.cls_branches[lvl](hs[lvl])             
            # 调用当前轮回归分支，预测边界框偏移量
            tmp = self.reg_branches[lvl](hs[lvl])                      
            # 合并参考点与偏移量：生成最终边界框坐标
            if reference.shape[-1] == 4:
                # 参考点为4维（cx, cy, w, h）：全量叠加偏移
                tmp += reference
            else:
                # 参考点为2维（cx, cy）：仅对中心坐标叠加偏移
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            # 对边界框坐标执行sigmoid：归一化到[0,1]范围（相对于图像尺寸）
            outputs_coord = tmp.sigmoid()
            # 收集当前轮的预测结果
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # 将多轮预测结果堆叠为tensor（维度：[轮次, 批次, 查询数, 类别数/4]）
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        # 返回多轮类别分数、边界框，及编码器topk提议的分数和锚点
        return outputs_classes, outputs_coords, topk_score, topk_anchor

    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_topk_scores,
             enc_topk_anchors,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             dn_meta=None,
             gt_bboxes_ignore=None):
        """
        计算总损失：包含编码器损失、解码器匹配损失、解码器去噪损失
        Args:
            all_cls_scores: 解码器所有轮次的类别分数
            all_bbox_preds: 解码器所有轮次的边界框预测
            enc_topk_scores: 编码器topk提议的类别分数
            enc_topk_anchors: 编码器topk提议的边界框
            gt_bboxes_list: 真实边界框列表
            gt_labels_list: 真实类别标签列表
            img_metas: 图像元信息列表
            dn_meta: 去噪元信息（含去噪查询的分组数、填充数等）
            gt_bboxes_ignore: 需忽略的真实边界框列表
        Returns:
            损失字典，包含所有损失项
        """
        # DINO不支持忽略边界框，故断言gt_bboxes_ignore必须为None
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__}仅支持gt_bboxes_ignore设为None.'

        # 初始化损失字典，用于存储各类损失
        loss_dict = dict()

        # 分离去噪查询输出和匹配查询输出（去噪输出用于计算去噪损失，匹配输出用于常规损失）
        all_cls_scores, all_bbox_preds, dn_cls_scores, dn_bbox_preds = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta)

        # 计算编码器损失（两阶段第一阶段：基于topk提议）
        if enc_topk_scores is not None:
            # 调用loss_single计算编码器的分类损失、边界框损失、IoU损失
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_topk_scores, enc_topk_anchors,
                                 gt_bboxes_list, gt_labels_list,
                                 img_metas, gt_bboxes_ignore)

            # 将编码器损失加入损失字典
            loss_dict['interm_loss_cls'] = enc_loss_cls
            loss_dict['interm_loss_bbox'] = enc_losses_bbox
            loss_dict['interm_loss_iou'] = enc_losses_iou

        # 计算解码器匹配损失（所有轮次，基于匹配查询输出）
        num_dec_layers = len(all_cls_scores)
        # 复制真实标签和配置，适配多轮解码器输出（每轮使用相同的真实标签）
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # 调用multi_apply并行计算每轮的损失
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        # 将最后一轮解码器的损失作为主损失，加入损失字典
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        # 将前几轮解码器的损失作为辅助损失，加入损失字典（命名格式：d{轮次}.loss_xxx）
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        # 计算解码器去噪损失（若有去噪输出）
        if dn_cls_scores is not None:
            # 复制去噪元信息，适配多轮解码器输出
            dn_meta = [dn_meta for _ in img_metas]
            # 调用loss_dn计算所有轮次的去噪损失
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
                img_metas, dn_meta)
            # 将最后一轮去噪损失作为主去噪损失，加入损失字典
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            # 将前几轮去噪损失作为辅助去噪损失，加入损失字典
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1

        # 返回最终的损失字典
        return loss_dict

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                gt_labels_list, img_metas, dn_meta):
        """
        计算解码器去噪损失：调用multi_apply并行处理多轮去噪输出
        Args:
            dn_cls_scores: 解码器所有轮次的去噪类别分数
            dn_bbox_preds: 解码器所有轮次的去噪边界框预测
            gt_bboxes_list: 真实边界框列表
            gt_labels_list: 真实类别标签列表
            img_metas: 图像元信息列表
            dn_meta: 去噪元信息列表（适配多轮输出）
        Returns:
            多轮去噪损失的列表（分类损失、边界框损失、IoU损失）
        """
        num_dec_layers = len(dn_cls_scores)
        # 复制真实标签和配置，适配多轮去噪输出
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        # 调用multi_apply并行计算每轮的去噪损失
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds,
                           all_gt_bboxes_list, all_gt_labels_list,
                           img_metas_list, dn_meta_list)

    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                       gt_labels_list, img_metas, dn_meta):
        """
        计算单轮去噪损失：生成去噪查询的正负样本标签→计算分类/边界框/IoU损失
        Args:
            dn_cls_scores: 单轮去噪类别分数
            dn_bbox_preds: 单轮去噪边界框预测
            gt_bboxes_list: 真实边界框列表
            gt_labels_list: 真实类别标签列表
            img_metas: 图像元信息列表
            dn_meta: 去噪元信息（单轮）
        Returns:
            单轮去噪的分类损失、边界框损失、IoU损失
        """
        # 获取批次大小
        num_imgs = dn_cls_scores.size(0)
        # 按图像拆分边界框预测（每个元素对应单张图像的预测）
        bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        # 调用get_dn_target生成去噪查询的正负样本标签和权重
        cls_reg_targets = self.get_dn_target(bbox_preds_list, gt_bboxes_list,
                                             gt_labels_list, img_metas,
                                             dn_meta)
        # 解包目标变量：类别标签、类别权重、边界框目标、边界框权重、正负样本数量
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 将所有图像的目标变量拼接为单tensor（便于批量计算损失）
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # 计算分类损失
        # 将类别分数展平为（批次×查询数，类别数）的形状
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # 构建分类损失的平均因子（正负样本加权，匹配官方DINO实现）
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        # 分布式训练中同步平均因子
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        # 确保平均因子不小于1（避免除以0）
        cls_avg_factor = max(cls_avg_factor, 1)

        # 计算分类损失（调用父类的loss_cls方法，如交叉熵损失）
        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            # 无样本时返回0损失（避免tensor形状错误）
            loss_cls = torch.zeros(  # TODO: 可优化0损失的返回方式
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # 计算边界框损失和IoU损失的平均因子（正样本数量，分布式同步）
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # 构建图像尺寸因子（用于将归一化边界框还原为像素坐标）
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            # 获取单张图像的原始尺寸
            img_h, img_w, _ = img_meta['img_shape']
            # 生成尺寸因子（[img_w, img_h, img_w, img_h]），适配边界框的4个维度
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        # 拼接所有图像的尺寸因子
        factors = torch.cat(factors, 0)

        # DETR系列算法中，边界框预测的是相对于图像的归一化坐标（cxcywh格式），
        # 计算IoU损失时需先还原为像素坐标（xyxy格式）
        # 展平边界框预测和目标
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        # 将预测和目标的边界框格式从cxcywh转为xyxy，并乘以尺寸因子还原为像素坐标
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # 计算IoU损失（调用父类的loss_iou方法，如GIoU损失）
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # 计算边界框L1损失（调用父类的loss_bbox方法）
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        # 返回单轮去噪的三类损失
        return loss_cls, loss_bbox, loss_iou

    def get_dn_target(self, dn_bbox_preds_list, gt_bboxes_list, gt_labels_list,
                      img_metas, dn_meta):
        """
        生成去噪查询的目标（正负样本标签和权重）：调用multi_apply并行处理每张图像
        Args:
            dn_bbox_preds_list: 去噪边界框预测列表（每个元素对应单张图像）
            gt_bboxes_list: 真实边界框列表
            gt_labels_list: 真实类别标签列表
            img_metas: 图像元信息列表
            dn_meta: 去噪元信息列表
        Returns:
            目标变量列表（类别标签、类别权重、边界框目标、边界框权重、正负样本索引、正负样本数量）
        """
        # 调用multi_apply并行计算每张图像的去噪目标
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, img_metas, dn_meta)
        # 统计所有图像的正负样本总数
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # 返回目标变量列表和正负样本数量
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, gt_bboxes, gt_labels,
                              img_meta, dn_meta):
        """
        生成单张图像的去噪查询目标：划分正负样本→分配真实标签和边界框
        Args:
            dn_bbox_pred: 单张图像的去噪边界框预测
            gt_bboxes: 单张图像的真实边界框
            gt_labels: 单张图像的真实类别标签
            img_meta: 单张图像的元信息
            dn_meta: 单张图像的去噪元信息
        Returns:
            单张图像的目标变量（类别标签、类别权重、边界框目标、边界框权重、正负样本索引）
        """
        # 从去噪元信息中获取分组数和去噪查询总数量
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        # 断言去噪查询总数能被分组数整除（每组去噪查询数量相同）
        assert pad_size % num_groups == 0
        # 计算每组去噪查询的数量
        single_pad = pad_size // num_groups
        # 获取单张图像的去噪查询总数
        num_bboxes = dn_bbox_pred.size(0)

        # 划分正负样本索引（基于是否有真实目标）
        if len(gt_labels) > 0:
            # 生成真实标签的索引（0~len(gt_labels)-1）
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            # 重复索引num_groups次（每组去噪查询对应相同的真实标签）
            t = t.unsqueeze(0).repeat(num_groups, 1)
            # 展平真实标签索引，用于分配正样本的真实类别
            pos_assigned_gt_inds = t.flatten()
            # 计算正样本查询的索引：每组的前len(gt_labels)个查询为正样本
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            # 无真实目标时，正负样本索引均为空
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
        # 计算负样本查询的索引：每组的中间len(gt_labels)个查询为负样本
        neg_inds = pos_inds + single_pad // 2

        # 生成分类标签：默认设为背景类（num_classes），正样本分配真实类别
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        # 生成分类标签权重：默认1（无忽略样本）
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 生成边界框目标：默认0，正样本分配真实边界框
        bbox_targets = torch.zeros_like(dn_bbox_pred)
        # 生成边界框权重：默认0，正样本设为1（仅正样本计算边界框损失）
        bbox_weights = torch.zeros_like(dn_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # 获取单张图像的原始尺寸
        img_h, img_w, _ = img_meta['img_shape']

        # DETR系列算法中，边界框目标需为相对于图像的归一化cxcywh格式，
        # 故需将原始真实边界框（xyxy格式，像素坐标）转换为归一化cxcywh格式
        # 生成尺寸因子（用于归一化）
        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)
        # 将真实边界框除以尺寸因子，归一化到[0,1]范围
        gt_bboxes_normalized = gt_bboxes / factor
        # 将真实边界框格式从xyxy转为cxcywh
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        # 为正样本分配真实边界框目标（重复num_groups次，适配多组去噪查询）
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        # 返回单张图像的目标变量和正负样本索引
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta):
        """
        静态方法：分离解码器输出中的“去噪查询输出”和“匹配查询输出”
        Args:
            all_cls_scores: 解码器所有轮次的类别分数
            all_bbox_preds: 解码器所有轮次的边界框预测
            dn_meta: 去噪元信息（含去噪查询的填充数pad_size）
        Returns:
            匹配查询输出（类别分数、边界框）和去噪查询输出（类别分数、边界框）
        """
        if dn_meta is not None:
            # 从去噪元信息中获取去噪查询的填充数（前pad_size个查询为去噪查询）
            denoising_cls_scores = all_cls_scores[:, :, :
                                                  dn_meta['pad_size'], :]
            denoising_bbox_preds = all_bbox_preds[:, :, :
                                                  dn_meta['pad_size'], :]
            # pad_size之后的查询为匹配查询
            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
            matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
        else:
            # 无去噪时，所有输出均为匹配查询输出，去噪输出设为None
            denoising_cls_scores = None
            denoising_bbox_preds = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
        # 返回匹配查询输出和去噪查询输出
        return (matching_cls_scores, matching_bbox_preds, denoising_cls_scores,
                denoising_bbox_preds)





# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply,
#                         reduce_mean)
# from mmdet.models.utils.transformer import inverse_sigmoid
# from mmdet.models.builder import HEADS

# from .mmdet_detr_head import DeformableDETRHead
# from .query_denoising import build_dn_generator


# @HEADS.register_module()
# class DINOHead(DeformableDETRHead):

#     def __init__(self,
#                  *args,
#                  num_query=100,
#                  dn_cfg=None,
#                  transformer=None,
#                  **kwargs):

#         if 'two_stage_num_proposals' in transformer:
#             assert transformer['two_stage_num_proposals'] == num_query, \
#                 'two_stage_num_proposals must be equal to num_query for DINO'
#         else:
#             transformer['two_stage_num_proposals'] = num_query
#         super(DINOHead, self).__init__(
#             *args, num_query=num_query, transformer=transformer, **kwargs)

#         assert self.as_two_stage, \
#             'as_two_stage must be True for DINO'
#         assert self.with_box_refine, \
#             'with_box_refine must be True for DINO'
#         self._init_layers()
#         self.init_denoising(dn_cfg)

#     def _init_layers(self):
#         super()._init_layers()
#         # NOTE The original repo of DINO set the num_embeddings 92 for coco,
#         # 91 (0~90) of which represents target classes and the 92 (91)
#         # indicates [Unknown] class. However, the embedding of unknown class
#         # is not used in the original DINO
#         self.label_embedding = nn.Embedding(self.num_classes,
#                                             self.embed_dims)

#     def init_denoising(self, dn_cfg):
#         if dn_cfg is not None:
#             dn_cfg['num_classes'] = self.num_classes
#             dn_cfg['num_queries'] = self.num_query
#             dn_cfg['hidden_dim'] = self.embed_dims
#         self.dn_generator = build_dn_generator(dn_cfg)

#     def forward_train(self,
#                       mlvl_feats,
#                       img_metas,
#                       gt_bboxes,
#                       gt_labels=None,
#                       gt_bboxes_ignore=None,
#                       shared_encoder=None,
#                       proposal_cfg=None,
#                       **kwargs):
#         assert proposal_cfg is None, '"proposal_cfg" must be None'
#         assert self.dn_generator is not None, '"dn_cfg" must be set'
#         dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
#             self.dn_generator(gt_bboxes, gt_labels,
#                               self.label_embedding, img_metas)
#         outs = self(shared_encoder, mlvl_feats, img_metas,
#                     dn_label_query, dn_bbox_query, attn_mask)
#         if gt_labels is None:
#             loss_inputs = outs + (gt_bboxes, img_metas, dn_meta)
#         else:
#             loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, dn_meta)
#         losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
#         return losses

#     def simple_test(self, feats, img_metas, shared_encoder=None, rescale=False):
#         outs = self.forward(shared_encoder, feats, img_metas)
#         results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
#         return results_list

#     def forward(self,
#                 encoder,
#                 mlvl_feats,
#                 img_metas,
#                 dn_label_query=None,
#                 dn_bbox_query=None,
#                 attn_mask=None):
#         batch_size = mlvl_feats[0].size(0)
#         input_img_h, input_img_w = img_metas[0]['batch_input_shape']
#         img_masks = mlvl_feats[0].new_ones(
#             (batch_size, input_img_h, input_img_w))
#         for img_id in range(batch_size):
#             img_h, img_w, _ = img_metas[img_id]['img_shape']
#             img_masks[img_id, :img_h, :img_w] = 0

#         mlvl_masks = []
#         mlvl_positional_encodings = []
#         for feat in mlvl_feats:
#             mlvl_masks.append(
#                 F.interpolate(img_masks[None],
#                               size=feat.shape[-2:]).to(torch.bool).squeeze(0))
#             mlvl_positional_encodings.append(
#                 self.positional_encoding(mlvl_masks[-1]))

#         query_embeds = None
#         hs, inter_references, topk_score, topk_anchor = \
#             self.transformer(
#                 mlvl_feats,
#                 mlvl_masks,
#                 query_embeds,
#                 mlvl_positional_encodings,
#                 dn_label_query,
#                 dn_bbox_query,
#                 attn_mask,
#                 encoder,  # Modification
#                 reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
#                 cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
#             )
#         hs = hs.permute(0, 2, 1, 3)

#         if dn_label_query is not None and dn_label_query.size(1) == 0:
#             # NOTE: If there is no target in the image, the parameters of
#             # label_embedding won't be used in producing loss, which raises
#             # RuntimeError when using distributed mode.
#             hs[0] += self.label_embedding.weight[0, 0] * 0.0

#         outputs_classes = []
#         outputs_coords = []

#         for lvl in range(hs.shape[0]):
#             reference = inter_references[lvl]
#             reference = inverse_sigmoid(reference, eps=1e-3)
#             outputs_class = self.cls_branches[lvl](hs[lvl])
#             tmp = self.reg_branches[lvl](hs[lvl])
#             if reference.shape[-1] == 4:
#                 tmp += reference
#             else:
#                 assert reference.shape[-1] == 2
#                 tmp[..., :2] += reference
#             outputs_coord = tmp.sigmoid()
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)

#         outputs_classes = torch.stack(outputs_classes)
#         outputs_coords = torch.stack(outputs_coords)

#         return outputs_classes, outputs_coords, topk_score, topk_anchor

#     def loss(self,
#              all_cls_scores,
#              all_bbox_preds,
#              enc_topk_scores,
#              enc_topk_anchors,
#              gt_bboxes_list,
#              gt_labels_list,
#              img_metas,
#              dn_meta=None,
#              gt_bboxes_ignore=None):
#         assert gt_bboxes_ignore is None, \
#             f'{self.__class__.__name__} only supports ' \
#             f'for gt_bboxes_ignore setting to None.'

#         loss_dict = dict()

#         # extract denoising and matching part of outputs
#         all_cls_scores, all_bbox_preds, dn_cls_scores, dn_bbox_preds = \
#             self.extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta)

#         if enc_topk_scores is not None:
#             # calculate loss from encode feature maps
#             # NOTE The DeformDETR calculate binary cls loss
#             # for all encoder embeddings, while DINO calculate
#             # multi-class loss for topk embeddings.
#             enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
#                 self.loss_single(enc_topk_scores, enc_topk_anchors,
#                                  gt_bboxes_list, gt_labels_list,
#                                  img_metas, gt_bboxes_ignore)

#             # collate loss from encode feature maps
#             loss_dict['interm_loss_cls'] = enc_loss_cls
#             loss_dict['interm_loss_bbox'] = enc_losses_bbox
#             loss_dict['interm_loss_iou'] = enc_losses_iou

#         # calculate loss from all decoder layers
#         num_dec_layers = len(all_cls_scores)
#         all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_bboxes_ignore_list = [
#             gt_bboxes_ignore for _ in range(num_dec_layers)
#         ]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]
#         losses_cls, losses_bbox, losses_iou = multi_apply(
#             self.loss_single, all_cls_scores, all_bbox_preds,
#             all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
#             all_gt_bboxes_ignore_list)

#         # collate loss from the last decoder layer
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_bbox'] = losses_bbox[-1]
#         loss_dict['loss_iou'] = losses_iou[-1]

#         # collate loss from other decoder layers
#         num_dec_layer = 0
#         for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
#                                                        losses_bbox[:-1],
#                                                        losses_iou[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
#             loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
#             num_dec_layer += 1

#         if dn_cls_scores is not None:
#             # calculate denoising loss from all decoder layers
#             dn_meta = [dn_meta for _ in img_metas]
#             dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
#                 dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
#                 img_metas, dn_meta)
#             # collate denoising loss
#             loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
#             loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
#             loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
#             num_dec_layer = 0
#             for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
#                     dn_losses_cls[:-1], dn_losses_bbox[:-1],
#                     dn_losses_iou[:-1]):
#                 loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
#                 loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
#                 loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
#                 num_dec_layer += 1

#         return loss_dict

#     def loss_dn(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
#                 gt_labels_list, img_metas, dn_meta):
#         num_dec_layers = len(dn_cls_scores)
#         all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]
#         dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
#         return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds,
#                            all_gt_bboxes_list, all_gt_labels_list,
#                            img_metas_list, dn_meta_list)

#     def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
#                        gt_labels_list, img_metas, dn_meta):
#         num_imgs = dn_cls_scores.size(0)
#         bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
#         cls_reg_targets = self.get_dn_target(bbox_preds_list, gt_bboxes_list,
#                                              gt_labels_list, img_metas,
#                                              dn_meta)
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          num_total_pos, num_total_neg) = cls_reg_targets
#         labels = torch.cat(labels_list, 0)
#         label_weights = torch.cat(label_weights_list, 0)
#         bbox_targets = torch.cat(bbox_targets_list, 0)
#         bbox_weights = torch.cat(bbox_weights_list, 0)

#         # classification loss
#         cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
#         # construct weighted avg_factor to match with the official DETR repo
#         cls_avg_factor = \
#             num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
#         if self.sync_cls_avg_factor:
#             cls_avg_factor = reduce_mean(
#                 cls_scores.new_tensor([cls_avg_factor]))
#         cls_avg_factor = max(cls_avg_factor, 1)

#         if len(cls_scores) > 0:
#             loss_cls = self.loss_cls(
#                 cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
#         else:
#             loss_cls = torch.zeros(  # TODO: How to better return zero loss
#                 1,
#                 dtype=cls_scores.dtype,
#                 device=cls_scores.device)

#         # Compute the average number of gt boxes across all gpus, for
#         # normalization purposes
#         num_total_pos = loss_cls.new_tensor([num_total_pos])
#         num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

#         # construct factors used for rescale bboxes
#         factors = []
#         for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
#             img_h, img_w, _ = img_meta['img_shape']
#             factor = bbox_pred.new_tensor([img_w, img_h, img_w,
#                                            img_h]).unsqueeze(0).repeat(
#                                                bbox_pred.size(0), 1)
#             factors.append(factor)
#         factors = torch.cat(factors, 0)

#         # DETR regress the relative position of boxes (cxcywh) in the image,
#         # thus the learning target is normalized by the image size. So here
#         # we need to re-scale them for calculating IoU loss
#         bbox_preds = dn_bbox_preds.reshape(-1, 4)
#         bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
#         bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

#         # regression IoU loss, defaultly GIoU loss
#         loss_iou = self.loss_iou(
#             bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

#         # regression L1 loss
#         loss_bbox = self.loss_bbox(
#             bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
#         return loss_cls, loss_bbox, loss_iou

#     def get_dn_target(self, dn_bbox_preds_list, gt_bboxes_list, gt_labels_list,
#                       img_metas, dn_meta):
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          pos_inds_list,
#          neg_inds_list) = multi_apply(self._get_dn_target_single,
#                                       dn_bbox_preds_list, gt_bboxes_list,
#                                       gt_labels_list, img_metas, dn_meta)
#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, bbox_targets_list,
#                 bbox_weights_list, num_total_pos, num_total_neg)

#     def _get_dn_target_single(self, dn_bbox_pred, gt_bboxes, gt_labels,
#                               img_meta, dn_meta):
#         num_groups = dn_meta['num_dn_group']
#         pad_size = dn_meta['pad_size']
#         assert pad_size % num_groups == 0
#         single_pad = pad_size // num_groups
#         num_bboxes = dn_bbox_pred.size(0)

#         if len(gt_labels) > 0:
#             t = torch.range(0, len(gt_labels) - 1).long().cuda()
#             t = t.unsqueeze(0).repeat(num_groups, 1)
#             pos_assigned_gt_inds = t.flatten()
#             pos_inds = (torch.tensor(range(num_groups)) *
#                         single_pad).long().cuda().unsqueeze(1) + t
#             pos_inds = pos_inds.flatten()
#         else:
#             pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
#         neg_inds = pos_inds + single_pad // 2

#         # label targets
#         labels = gt_bboxes.new_full((num_bboxes, ),
#                                     self.num_classes,
#                                     dtype=torch.long)
#         labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
#         label_weights = gt_bboxes.new_ones(num_bboxes)

#         # bbox targets
#         bbox_targets = torch.zeros_like(dn_bbox_pred)
#         bbox_weights = torch.zeros_like(dn_bbox_pred)
#         bbox_weights[pos_inds] = 1.0
#         img_h, img_w, _ = img_meta['img_shape']

#         # DETR regress the relative position of boxes (cxcywh) in the image.
#         # Thus the learning target should be normalized by the image size, also
#         # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
#         factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
#                                           img_h]).unsqueeze(0)
#         gt_bboxes_normalized = gt_bboxes / factor
#         gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
#         bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

#         return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
#                 neg_inds)

#     @staticmethod
#     def extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta):
#         if dn_meta is not None:
#             denoising_cls_scores = all_cls_scores[:, :, :
#                                                   dn_meta['pad_size'], :]
#             denoising_bbox_preds = all_bbox_preds[:, :, :
#                                                   dn_meta['pad_size'], :]
#             matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
#             matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
#         else:
#             denoising_cls_scores = None
#             denoising_bbox_preds = None
#             matching_cls_scores = all_cls_scores
#             matching_bbox_preds = all_bbox_preds
#         return (matching_cls_scores, matching_bbox_preds, denoising_cls_scores,
#                 denoising_bbox_preds)
