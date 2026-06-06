"""
Step 3: 模型 & DataLoader 修改说明
将离线预计算的实体特征注入 GeoRSCLIPModel 解码器。

本文件包含：
  A. DataLoader 的修改 patch（在 Dataset 类里加载 entity npz）
  B. GeoRSCLIPModel 的修改 patch（ECE 投影层 + 解码时注入实体特征）
  C. DecoderLayer2 修改 patch（解码器融合实体特征）

使用方式：将对应代码段复制到你的原始文件中替换即可。
每个 patch 都有 [原始代码] 和 [替换为] 标注。
"""

# =============================================================================
# ★ PATCH A：DataLoader —— 在 Dataset 中加载实体特征
# =============================================================================
#
# 文件：dataloader.py（你的 Dataset 类）
#
# [A-1] 在 __init__ 中，紧跟 att_loader 初始化之后，添加：
# ─────────────────────────────────────────────────────────
#
#   self.entity_loader = HybridLoader(self.opt.input_entity_dir, '.npz',
#                                     in_memory=self.data_in_memory)
#
# ─────────────────────────────────────────────────────────
# 说明：
#   opt.input_entity_dir = "data/UCM_entity" （Step 2 的输出目录）
#   文件名格式：{img_id}_entity.npz，key='feat'，shape=(M, C_clip)
#
#
# [A-2] 在 __getitem__ 中，紧跟 att_feat 加载之后，添加：
# ─────────────────────────────────────────────────────────
#
#   if getattr(self, 'entity_loader', None) is not None:
#       entity_feat = self.entity_loader.get(
#           str(self.info['images'][ix]['id']) + '_entity'
#       )                                                   # (M, C_clip)
#   else:
#       entity_feat = np.zeros((5, 1024), dtype='float32') # fallback
#
# ─────────────────────────────────────────────────────────
#
#
# [A-3] 修改 __getitem__ 的 return，把 entity_feat 加进去：
# ─────────────────────────────────────────────────────────
#
#   return (fc_feat, att_feat, entity_feat, seq, ix, it_pos_now, wrapped)
#
# ─────────────────────────────────────────────────────────
#
#
# [A-4] 在 collate_func 中，同步解包并 stack entity_feat：
# ─────────────────────────────────────────────────────────
#
#   # 在 collate_func 里，解包处改为：
#   for sample in batch:
#       tmp_fc, tmp_att, tmp_entity, tmp_seq, ix, it_pos_now, tmp_wrapped = sample
#       ...
#       entity_batch.append(tmp_entity)    # 新增
#
#   # 在组装 data 字典时添加：
#   data['entity_feats'] = np.stack(entity_batch)  # (B, M, C_clip)
#
#   # 并在最后的 tensor 转换中自动被覆盖（已有的那行 for k,v in data.items()）
#
# =============================================================================


# =============================================================================
# ★ PATCH B：GeoRSCLIPModel —— 添加 ECE 投影层，传递实体特征
# =============================================================================
#
# 文件：models/GeoRSCLIPModel.py
#
# [B-1] 新增 ECEProjection 类（放在 GeoRSCLIPModel 定义之前）：
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECEProjection(nn.Module):
    """
    将 CLIP 文本空间的实体特征（C_clip=1024）
    投影到解码器特征空间（d_model=512）。

    对应论文 PCI 模块中对 Fe 的处理：
      Fe = CLIPt(Pe)  →  Linear 投影  →  参与交叉注意力
    """

    def __init__(self, clip_dim: int = 1024, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, entity_feats: torch.Tensor) -> torch.Tensor:
        """
        entity_feats: (B, M, C_clip)
        返回:          (B, M, d_model)
        """
        return self.proj(entity_feats)


# ─────────────────────────────────────────────────────────
#
# [B-2] 在 GeoRSCLIPModel.__init__ 末尾添加投影层：
# ─────────────────────────────────────────────────────────
#
#   # ECE 投影层：将 CLIP 文本特征对齐到解码器维度
#   self.ece_proj = ECEProjection(
#       clip_dim=1024,          # GeoRSCLIP ViT-H-14 的输出维度
#       d_model=self.d_model,   # 512
#       dropout=self.dropout,
#   )
#
# ─────────────────────────────────────────────────────────
#
#
# [B-3] 修改 _prepare_feature，将实体特征一并返回：
# ─────────────────────────────────────────────────────────
#
#   def _prepare_feature(self, fc_feats, att_feats, att_masks, entity_feats=None):
#       att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
#           att_feats, att_masks
#       )
#       memory = self.model.encode(att_feats, att_masks)
#
#       # 投影实体特征
#       if entity_feats is not None:
#           entity_proj = self.ece_proj(entity_feats)   # (B, M, d_model)
#       else:
#           entity_proj = None
#
#       return fc_feats[..., :0], att_feats[..., :0], memory, att_masks, entity_proj
#
# ─────────────────────────────────────────────────────────
#
#
# [B-4] 修改 _forward，从 batch 中取出 entity_feats 并传入：
# ─────────────────────────────────────────────────────────
#
#   def _forward(self, fc_feats, att_feats, seq, att_masks=None, entity_feats=None):
#       if seq.ndim == 3:
#           seq = seq.reshape(-1, seq.shape[2])
#       att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
#           att_feats, att_masks, seq
#       )
#       # 投影实体特征
#       if entity_feats is not None:
#           entity_proj = self.ece_proj(entity_feats)   # (B, M, d_model)
#       else:
#           entity_proj = None
#
#       out = self.model(att_feats, seq, att_masks, seq_mask, entity_proj)
#       outputs = self.model.generator(out)
#       return outputs
#
# ─────────────────────────────────────────────────────────


# =============================================================================
# ★ PATCH C：EncoderDecoder & DecoderLayer2 —— 融合实体特征
# =============================================================================
#
# [C-1] 修改 EncoderDecoder.forward / decode，把 entity_proj 透传给 decoder2：
# ─────────────────────────────────────────────────────────
#
#   def forward(self, src, tgt, src_mask, tgt_mask, entity_proj=None):
#       encoder_out = self.encode(src, src_mask)
#       return self.decode(encoder_out, src_mask, tgt, tgt_mask, entity_proj)
#
#   def decode(self, memory, src_mask, tgt, tgt_mask, entity_proj=None):
#       encoder_out = self.decoder1(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
#       return self.decoder2(encoder_out, memory, src_mask, tgt_mask, entity_proj)
#
# ─────────────────────────────────────────────────────────
#
#
# [C-2] 修改 Decoder2.forward，透传 entity_proj：
# ─────────────────────────────────────────────────────────
#
#   def forward(self, x, memory, src_mask, tgt_mask, entity_proj=None):
#       for layer in self.layers:
#           x = layer(x, memory, src_mask, tgt_mask, entity_proj)
#       return self.norm(x)
#
# ─────────────────────────────────────────────────────────
#
#
# [C-3] 修改 DecoderLayer2.__init__，添加实体交叉注意力和融合权重：
# ─────────────────────────────────────────────────────────
#
#   # 在现有属性后追加：
#   self.entity_attn   = copy.deepcopy(attn)          # 实体提示交叉注意力
#   self.fc_alpha_ent  = nn.Linear(d_model * 2, d_model)  # 实体融合权重
#
#   注意：attn 是 MultiHeadedAttention 实例，需要在 make_model 里传进来，
#         或者直接在 DecoderLayer2.__init__ 里新建一个：
#
#   self.entity_attn  = MultiHeadedAttention(h=8, d_model=512, dropout=0.1)
#   self.fc_alpha_ent = nn.Linear(512 + 512, 512)
#
# ─────────────────────────────────────────────────────────
#
#
# [C-4] 完整替换 DecoderLayer2.forward（将实体融合嵌入现有流程）：
# ─────────────────────────────────────────────────────────

class DecoderLayer2WithECE(nn.Module):
    """
    在原 DecoderLayer2 基础上，
    额外引入实体提示交叉注意力（来自 ECE），
    并用动态 alpha 与视觉注意力结果融合。

    将此类替换原文件中的 DecoderLayer2 即可。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        import copy, numpy as np
        from .GeoRSCLIPModel import (
            SublayerConnection, LayerNorm, MultiHeadedAttention,
            PositionwiseFeedForward, ChannelAttention, SpatialAttention,
        )
        self.size        = size
        self.self_attn   = self_attn
        self.src_attn    = src_attn
        self.feed_forward = feed_forward
        self.sublayer    = nn.ModuleList([
            SublayerConnection(size, dropout) for _ in range(3)
        ])

        self.d_model      = 512
        self.mul_level_ffn = nn.Linear(self.d_model * 3, self.d_model)
        self.fc_alpha1    = nn.Linear(self.d_model + self.d_model, self.d_model)
        self.fc_alpha2    = nn.Linear(self.d_model + self.d_model, self.d_model)

        # ── 新增：实体提示交叉注意力 & 融合权重 ──────────────────
        self.entity_attn   = MultiHeadedAttention(h=8, d_model=self.d_model, dropout=dropout)
        self.fc_alpha_ent  = nn.Linear(self.d_model * 2, self.d_model)
        # ──────────────────────────────────────────────────────────

    def forward(self, x, memory, src_mask, tgt_mask, entity_proj=None):
        import numpy as np
        from .GeoRSCLIPModel import ChannelAttention, SpatialAttention

        m = memory

        # 1. 自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 2. 原有 Channel + Spatial 视觉注意力
        channel = ChannelAttention(512, 512).to(x.device)
        spatial  = SpatialAttention().to(x.device)
        yita     = nn.Parameter(torch.tensor(0.2)).to(x.device)

        Vm = self.mul_level_ffn(m) * yita + memory[..., 512 * 2: 512 * 3]
        Vm = Vm.view(Vm.shape[0], Vm.shape[2], 16, 16)
        Vc = channel(Vm).view(Vm.shape[0], -1, Vm.shape[1])
        Vs = spatial(Vm).view(Vm.shape[0], -1, Vm.shape[1])

        att1   = self.sublayer[1](x, lambda x: self.src_attn(x, Vc, Vc, src_mask))
        att2   = self.sublayer[1](x, lambda x: self.src_attn(x, Vs, Vs, src_mask))
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([x, att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([x, att2], -1)))
        x_vis  = (att1 * alpha1 + att2 * alpha2) / np.sqrt(2)   # (B, T, d)

        # 3. ── 新增：实体提示交叉注意力 ──────────────────────────
        if entity_proj is not None:
            # entity_proj: (B, M, d_model)
            att_ent   = self.entity_attn(x_vis, entity_proj, entity_proj)
            alpha_ent = torch.sigmoid(
                self.fc_alpha_ent(torch.cat([x_vis, att_ent], dim=-1))
            )
            # 动态融合视觉特征与实体特征
            x_vis = x_vis + alpha_ent * att_ent                  # 残差式融合
        # ──────────────────────────────────────────────────────────

        x = x_vis

        # 4. FFN
        return self.sublayer[2](x, self.feed_forward)


# =============================================================================
# ★ 训练脚本修改说明（train.py / captioning.py）
# =============================================================================
#
# 在 get_batch 之后，把 entity_feats 传给模型的 _forward：
#
#   data   = loader.get_batch('train')
#   fc_feats    = data['fc_feats'].cuda()
#   att_feats   = data['att_feats'].cuda()
#   labels      = data['labels'].cuda()
#   masks       = data['masks'].cuda()
#   entity_feats = data.get('entity_feats', None)
#   if entity_feats is not None:
#       entity_feats = entity_feats.cuda()
#
#   outputs = model(_forward)(fc_feats, att_feats, labels, entity_feats=entity_feats)
#
# =============================================================================


# =============================================================================
# ★ 使用流程总结
# =============================================================================
#
# 1. 准备实体列表
#    python build_entity_space.py \
#        --entity_path  data/entities_ucm.txt \
#        --output_path  data/entity_space.npz
#
# 2. 为每张图提取 Top-M 实体特征
#    python extract_entity_features.py \
#        --json_path   datasets/UCM_captions/dataset.json \
#        --images_dir  datasets/UCM_captions/imgs \
#        --entity_space data/entity_space.npz \
#        --output_dir   data/UCM_entity \
#        --top_m 5 \
#        --skip_existing
#
# 3. 修改 opt 配置，加入新路径
#    opt.input_entity_dir = "data/UCM_entity"
#
# 4. 按上述 PATCH A/B/C 修改 dataloader.py 和 GeoRSCLIPModel.py
#
# 5. 正常启动训练
#
# =============================================================================