from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
import copy
import math
import numpy as np
from .AttModel import pack_wrapper, AttModel


# ══════════════════════════════════════════════════════════════════
# 基础模块（与原始代码完全一致，不做任何改动）
# ══════════════════════════════════════════════════════════════════

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.add_pooling = lambda x: self.avg_pool(x)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_pre = self.add_pooling(x)
        att = self.sigmoid(self.relu(self.conv(x_pre)))
        return att * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.ave_pooling = lambda x: torch.mean(x, dim=1, keepdim=True)
        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        apa_att = self.sigmoid(self.conv(self.ave_pooling(x)))
        return apa_att * x


# ══════════════════════════════════════════════════════════════════
# ★ 新增：ECEProjection — CLIP文本特征投影层
# ══════════════════════════════════════════════════════════════════
class ECEProjection(nn.Module):
    """
    将 CLIP 文本空间实体特征 (1024) 投影到解码器维度 (512)。
    使用 GELU 避免 ReLU 对归一化特征的负值截断。
    """
    def __init__(self, clip_dim=1024, d_model=512, dropout=0.1):
        super(ECEProjection, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, entity_feats):
        # entity_feats: (B, M, 1024) → (B, M, 512)
        return self.proj(entity_feats)


# ══════════════════════════════════════════════════════════════════
# ★ 新增：PCIModule — 提示驱动跨模态交互
# 论文 Section III-D：以实体文本特征为 Query，视觉特征为 Key/Value
# 输出跨模态实体语义特征 Fcross_e，已对齐到视觉语义空间
# ══════════════════════════════════════════════════════════════════
class PCIModule(nn.Module):
    """
    Prompt-driven Cross-modal Interaction Module

    输入:
        Fe   : (B, M, d_model)  实体提示文本特征（ECEProjection 输出）
        Ven  : (B, N, d_model)  多尺度视觉特征（编码后）
    输出:
        Fcross_e: (B, M, d_model)  跨模态实体语义特征
    """
    def __init__(self, d_model=512, ven_dim=None, n_heads=8, dropout=0.1):
        super(PCIModule, self).__init__()
        ven_dim = ven_dim or d_model

        # Fe (512) 作为 Q，Ven (1536) 作为 K/V
        # 需要先把 Ven 投影到 d_model，否则 MHA 维度不匹配
        self.ven_proj = nn.Linear(ven_dim, d_model) if ven_dim != d_model else nn.Identity()
        # 交叉注意力：Fe 为 Q，Ven 为 K/V
        self.cross_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Fe, Ven):
        """
        Fe : (B, M, d_model)
        Ven: (B, N, d_model)
        """
        # 先把 Ven 从 1536 投影到 512
        Ven_proj = self.ven_proj(Ven)  # (B, N, 512)

        # 交叉注意力：实体文本特征 attend 视觉特征
        # 等价于论文公式(8)：MHA(Fs/Fe, Ven, Ven)
        attn_out = self.cross_attn(Fe, Ven_proj, Ven_proj)  # (B, M, 512)
        Fe = self.norm1(Fe + self.dropout1(attn_out))       # 残差 + LN

        # FFN
        ffn_out = self.ffn(Fe)
        Fcross_e = self.norm2(Fe + self.dropout2(ffn_out))  # 残差 + LN

        return Fcross_e                                     # (B, M, d_model)


# ══════════════════════════════════════════════════════════════════
# Decoder 层（原始 HAT 结构 + ECE/PCI 融合，改动最小化）
# ══════════════════════════════════════════════════════════════════
class Decoder1(nn.Module):
    def __init__(self, layer, N):
        super(Decoder1, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.count = 1

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer1(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer1, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder2(nn.Module):
    def __init__(self, layer, N):
        super(Decoder2, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, fcross_e=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, src_mask, tgt_mask, fcross_e, layer_idx=i)
        return self.norm(x)


class DecoderLayer2(nn.Module):
    """
    原始 HAT DecoderLayer2，在 FFN 之前加入
    跨模态实体特征 Fcross_e 的融合（已由 PCI 对齐）。

    关键改动：
      - 接收 fcross_e 而非原始 entity_proj
      - fcross_e 已在视觉语义空间，可直接与视觉注意力结果融合
      - 用 sublayer[3] 保证残差 + dropout + LayerNorm
      - fc_alpha_ent bias 初始化为 -3，训练初期融合权重接近0
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer2, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # sublayer[3] 专门给实体注意力用
        self.sublayer = clones(SublayerConnection(size, dropout), 4)

        self.d_model = 512
        self.mul_level_ffn = nn.Linear(self.d_model * 3, self.d_model)
        self.fc_alpha1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc_alpha2 = nn.Linear(self.d_model * 2, self.d_model)

        # ── ECE+PCI 融合：实体交叉注意力 & 动态融合权重 ────────────
        self.entity_attn  = copy.deepcopy(src_attn)
        self.fc_alpha_ent = nn.Linear(self.d_model * 2, self.d_model)
        # 初始化：训练初期融合权重接近0，不干扰已收敛的视觉路径
        nn.init.zeros_(self.fc_alpha_ent.weight)
        nn.init.constant_(self.fc_alpha_ent.bias, -3.0)   # sigmoid(-3) ≈ 0.05
        # ──────────────────────────────────────────────────────────

        # Channel / Spatial Attention 改为模块成员，避免每次 forward 重建
        self.channel_att = ChannelAttention(512, 512)
        self.spatial_att = SpatialAttention()
        self.yita = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, memory, src_mask, tgt_mask, fcross_e=None, layer_idx=0):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # ── 原始 HAT：Channel + Spatial 视觉注意力 ────────────────
        Vm = self.mul_level_ffn(m) * self.yita + memory[..., 512 * 2:512 * 3]
        Vm = Vm.view(Vm.shape[0], Vm.shape[2], 16, 16)
        Vc = self.channel_att(Vm).view(Vm.shape[0], -1, Vm.shape[1])
        Vs = self.spatial_att(Vm).view(Vm.shape[0], -1, Vm.shape[1])
        att1   = self.sublayer[1](x, lambda x: self.src_attn(x, Vc, Vc, src_mask))
        att2   = self.sublayer[1](x, lambda x: self.src_attn(x, Vs, Vs, src_mask))
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([x, att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([x, att2], -1)))
        x = (att1 * alpha1 + att2 * alpha2) / np.sqrt(2)
        # ─────────────────────────────────────────────────────────

        # ── ECE+PCI：跨模态实体特征融合 ──────────────────────────
        # fcross_e: (B, M, d_model)，已由 PCI 对齐到视觉语义空间
        # 用 sublayer[3] 提供残差 + dropout + LayerNorm 保护
        if fcross_e is not None and layer_idx >= 3:
            def entity_fusion(x):
                att_ent   = self.entity_attn(x, fcross_e, fcross_e)
                alpha_ent = torch.sigmoid(
                    self.fc_alpha_ent(torch.cat([x, att_ent], dim=-1))
                )
                return alpha_ent * att_ent
            x = self.sublayer[3](x, entity_fusion)
        # ─────────────────────────────────────────────────────────

        return self.sublayer[2](x, self.feed_forward)


# ══════════════════════════════════════════════════════════════════
# EncoderDecoder：透传 fcross_e
# ══════════════════════════════════════════════════════════════════
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder1, decoder2, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder  = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, fcross_e=None):
        encoder_out = self.encode(src, src_mask)
        return self.decode(encoder_out, src_mask, tgt, tgt_mask, fcross_e)

    def encode(self, src, src_mask):
        enc_emb1 = self.src_embed(src[..., 0 * 512:1 * 512])
        enc_emb2 = self.src_embed(src[..., 1 * 512:2 * 512])
        enc_emb3 = self.src_embed(src[..., 2 * 512:3 * 512])
        enc_emb = torch.cat([enc_emb1, enc_emb2, enc_emb3], dim=-1)
        return self.encoder(enc_emb, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, fcross_e=None):
        encoder_out = self.decoder1(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return self.decoder2(encoder_out, memory, src_mask, tgt_mask, fcross_e)


# ══════════════════════════════════════════════════════════════════
# ★ 主模型：RSCLIP_ECE_PCI
# ══════════════════════════════════════════════════════════════════
class RSCLIP_ECE(AttModel):
    """
    在原始 GeoRSCLIPModel (HAT+PTE) 基础上，增加：
      1. ECEProjection：CLIP 实体文本特征投影
      2. PCIModule：实体文本特征 attend 视觉特征，生成跨模态对齐特征 Fcross_e
      3. DecoderLayer2 融合 Fcross_e：用动态 alpha 残差融合

    数据流：
      entity_feats (B,M,1024)
           ↓  ECEProjection
      Fe   (B,M,512)
           ↓  PCIModule (Fe作为Q, Ven作为K/V)
      Fcross_e (B,M,512)          ← 已对齐到视觉语义空间
           ↓
      DecoderLayer2 融合           ← 不再有模态错位问题
    """

    def make_model(self, src_vocab, tgt_vocab, N_enc, N_dec, d_model, d_ff, h, dropout):
        c    = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff   = PositionwiseFeedForward(d_model, d_ff, dropout)
        pos  = PositionalEncoding(d_model, dropout)

        model = EncoderDecoder(
            lambda x, y: x,
            Decoder1(DecoderLayer1(d_model, c(attn), c(ff), dropout), 4),
            Decoder2(DecoderLayer2(d_model, c(attn), c(attn), c(ff), dropout), 5),
            PositionalEncoding(d_model, dropout),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos)),
            Generator(d_model, tgt_vocab),
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(RSCLIP_ECE, self).__init__(opt)
        self.opt = opt

        self.N_enc   = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec   = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff    = getattr(opt, 'd_ff', opt.rnn_size)
        self.h       = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = clones(nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
            (nn.Linear(self.att_feat_size, self.d_model),
             nn.ReLU(),
             nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())
        )), 3)

        delattr(self, 'embed');   self.embed    = lambda x: x
        delattr(self, 'fc_embed'); self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab  = self.vocab_size + 1
        self.model = self.make_model(
            0, tgt_vocab,
            N_enc=self.N_enc, N_dec=self.N_dec,
            d_model=self.d_model, d_ff=self.d_ff,
            h=self.h, dropout=self.dropout,
        )

        # ── ECE 投影层 ─────────────────────────────────────────────
        self.ece_proj = ECEProjection(
            clip_dim=1024, d_model=self.d_model, dropout=self.dropout
        )

        # ── PCI 跨模态交互模块 ─────────────────────────────────────
        self.pci = PCIModule(
            d_model=self.d_model, ven_dim=self.d_model * 3, n_heads=self.h, dropout=self.dropout
        )
        # ──────────────────────────────────────────────────────────

    def logit(self, x):
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    # ── 核心：encode 后立即做 ECE+PCI，生成 Fcross_e ──────────────
    def _compute_fcross_e(self, memory, entity_feats):
        """
        memory      : (B, N, d_model)  视觉编码特征 Ven
        entity_feats: (B, M, 1024)     原始 CLIP 实体文本特征
        返回:
            fcross_e: (B, M, d_model)  跨模态对齐后的实体特征
        """
        if entity_feats is None:
            return None
        Fe       = self.ece_proj(entity_feats.float())  # (B, M, 512)
        fcross_e = self.pci(Fe, memory)                 # (B, M, 512)
        return fcross_e

    def _prepare_feature(self, fc_feats, att_feats, att_masks, entity_feats=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        # PCI 在 encode 之后立即执行，生成对齐特征并缓存
        fcross_e = self._compute_fcross_e(memory, entity_feats)
        self._fcross_e_cache = fcross_e                  # 供 beam search core() 使用

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats1 = pack_wrapper(self.att_embed[0], att_feats[..., 0 * 1280:1 * 1280], att_masks)
        att_feats2 = pack_wrapper(self.att_embed[1], att_feats[..., 1 * 1280:2 * 1280], att_masks)
        att_feats3 = pack_wrapper(self.att_embed[2], att_feats[..., 2 * 1280:3 * 1280], att_masks)
        att_feats  = torch.cat([att_feats1, att_feats2, att_feats3], dim=-1)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:, 0] = 1
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img, [att_feats, att_masks])
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, entity_feats=None):
        if seq.ndim == 3:
            seq = seq.reshape(-1, seq.shape[2])

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
            att_feats, att_masks, seq
        )

        # encode 得到视觉特征 memory (B, N, d_model)
        # 注意：此处 att_feats 已经过 seq_per_img 复制，memory 需要用原始 B 的 att_feats
        # 所以先取前 B 张做 encode，再计算 PCI
        B = entity_feats.shape[0] if entity_feats is not None else att_feats.shape[0]
        seq_per_img = att_feats.shape[0] // B

        # 用原始 B 张图的视觉特征做 encode（取前 B 张）
        att_feats_single = att_feats[:B]
        att_masks_single = att_masks[:B]
        memory_single    = self.model.encode(att_feats_single, att_masks_single)  # (B, N, d)

        # ECE + PCI：生成 Fcross_e
        fcross_e = self._compute_fcross_e(memory_single, entity_feats)  # (B, M, d)

        # 完整 memory（含 seq_per_img 复制）用于解码
        memory = self.model.encode(att_feats, att_masks)  # (B*seq_per_img, N, d)

        # fcross_e 按 seq_per_img 复制，与 memory batch 对齐
        if fcross_e is not None and seq_per_img > 1:
            fcross_e = fcross_e.unsqueeze(1).expand(
                -1, seq_per_img, -1, -1
            ).reshape(-1, fcross_e.shape[1], fcross_e.shape[2])  # (B*seq_per_img, M, d)

        out     = self.model.decode(memory, att_masks, seq, seq_mask, fcross_e)
        outputs = self.model.generator(out)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """beam search 逐步解码，从缓存取 fcross_e"""
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        fcross_e = getattr(self, '_fcross_e_cache', None)

        out = self.model.decode(
            memory, mask, ys,
            subsequent_mask(ys.size(1)).to(memory.device),
            fcross_e,
        )
        return out[:, -1], [ys.unsqueeze(0)]