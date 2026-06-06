"""
Step 2: Extract Per-Image Entity Features (ECE 离线预计算)
为数据集中每张图片：
  1. 加载已有 CLIP 视觉特征（复用 extract_features.py 提取的 npz）
     或重新用 CLIP 视觉编码器提取全局特征
  2. 与 entity_space.npz 中的文本特征计算余弦相似度
  3. 选 Top-M 实体，保存其对应文本特征 → {img_id}_entity.npz

输出格式：
  feat:     (M, C_clip)  float32   —— Top-M 实体的 L2 归一化文本特征
  indices:  (M,)         int32     —— 在实体空间中的下标（可选，方便 debug）
  scores:   (M,)         float32   —— 对应相似度得分（可选）

后续在 dataloader 里加载 {img_id}_entity.npz 即可。
"""

import os
import json
import argparse
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# ============================================================
# CONFIG —— 按你的实际路径修改
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_CKPT    = os.path.join(PROJECT_ROOT, "RS5M_ViT-H-14.pt")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOP_M        = 5      # 每张图选取的实体数量，可通过 --top_m 调整
# ============================================================


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────
def load_clip_model(ckpt_path: str, device: str):
    """加载 GeoRSCLIP (ViT-H-14)，返回 (model, preprocess)"""
    print(f"[CLIP] 加载模型: {ckpt_path}")
    result = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=ckpt_path
    )
    if isinstance(result, tuple):
        model = result[0]
    elif isinstance(result, dict):
        model = result["model"]
    else:
        raise TypeError(f"open_clip 返回类型未知: {type(result)}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # 与 extract_features.py 相同的预处理
    preprocess = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    print("[CLIP] 模型加载完成")
    return model, preprocess


# ──────────────────────────────────────────────
# 视觉全局特征提取（用于和实体文本做相似度）
# ──────────────────────────────────────────────
@torch.no_grad()
def extract_global_image_feat(model, preprocess, image_path: str, device: str) -> np.ndarray | None:
    """
    提取单张图片的 CLIP 全局视觉特征（CLS token 经 ln_post 后）。
    返回: (C_clip,) float32 numpy，L2 归一化
    """
    try:
        image = Image.open(image_path).convert("RGB")
        x = preprocess(image).unsqueeze(0).to(device)         # (1, 3, 224, 224)

        # 通过 ViT 所有层，取最后的全局特征
        x = model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = model.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)                                # (1+N, B, C)

        for block in model.visual.transformer.resblocks:
            x = block(x)

        x = x.permute(1, 0, 2)                                # (B, 1+N, C)
        cls_feat = model.visual.ln_post(x[:, 0, :])           # (B, C) —— CLS token
        if model.visual.proj is not None:
            cls_feat = cls_feat @ model.visual.proj            # (B, C_proj)

        cls_feat = cls_feat.float()
        cls_feat = cls_feat / (cls_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return cls_feat.squeeze(0).cpu().numpy()               # (C,)

    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return None


# ──────────────────────────────────────────────
# Top-M 实体选取
# ──────────────────────────────────────────────
def select_top_m_entities(
    image_feat: np.ndarray,       # (C,)
    entity_feats: np.ndarray,     # (E, C)
    top_m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算余弦相似度，返回 Top-M 实体的:
      selected_feats:   (M, C)   文本特征
      selected_indices: (M,)     在实体空间中的下标
      selected_scores:  (M,)     相似度得分
    """
    # 两者均已 L2 归一化，点积 = 余弦相似度
    scores  = entity_feats @ image_feat                        # (E,)
    top_idx = np.argsort(scores)[::-1][:top_m]                # (M,) 降序

    return (
        entity_feats[top_idx].astype(np.float32),             # (M, C)
        top_idx.astype(np.int32),
        scores[top_idx].astype(np.float32),
    )


# ──────────────────────────────────────────────
# 数据集遍历：支持 RSICD/UCM 和 NWPU 格式
# ──────────────────────────────────────────────
def iter_dataset(json_path: str, images_dir: str):
    """
    生成器：统一遍历不同数据集格式，
    每次 yield (img_id, img_path)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "images" in data:
        # RSICD / UCM 格式
        for img in data["images"]:
            img_id   = str(img.get("imgid", img.get("id", "")))
            filename = img.get("filename", "")
            if not filename:
                continue
            yield img_id, os.path.join(images_dir, filename)

    elif isinstance(data, dict):
        # NWPU 格式：{category: [{imgid, filename, ...}, ...]}
        for category, img_list in data.items():
            for img in img_list:
                img_id   = str(img.get("imgid", ""))
                filename = img.get("filename", "")
                if not filename:
                    continue
                yield img_id, os.path.join(images_dir, category, filename)

    else:
        raise ValueError(f"不支持的 JSON 格式，根类型为: {type(data)}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ECE: Extract Per-Image Entity Features")
    parser.add_argument("--json_path",        type=str, required=True,
                        help="数据集 json 路径（dataset.json 或 dataset_rsicd_modified.json 等）")
    parser.add_argument("--images_dir",       type=str, required=True,
                        help="图片根目录")
    parser.add_argument("--entity_space",     type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "entity_space.npz"),
                        help="Step 1 生成的 entity_space.npz 路径")
    parser.add_argument("--output_dir",       type=str, required=True,
                        help="每张图实体特征的输出目录，文件名为 {img_id}_entity.npz")
    parser.add_argument("--clip_ckpt",        type=str, default=CLIP_CKPT)
    parser.add_argument("--device",           type=str, default=DEVICE)
    parser.add_argument("--top_m",            type=int, default=TOP_M,
                        help="每张图选取的 Top-M 实体数量（论文默认 5）")
    parser.add_argument("--skip_existing",    action="store_true",
                        help="跳过已生成的文件（断点续跑）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载实体空间
    print(f"[Entity Space] 加载: {args.entity_space}")
    space      = np.load(args.entity_space, allow_pickle=True)
    ent_feats  = space["feat"].astype(np.float32)             # (E, C)
    ent_words  = space["entities"].tolist()                   # (E,) str list
    print(f"  实体数量 E={len(ent_words)}, 特征维度 C={ent_feats.shape[1]}")

    # 2. 加载 CLIP
    model, preprocess = load_clip_model(args.clip_ckpt, args.device)

    # 3. 遍历数据集
    items = list(iter_dataset(args.json_path, args.images_dir))
    print(f"[Dataset] 共 {len(items)} 张图片")

    skipped = 0
    errors  = 0

    for img_id, img_path in tqdm(items, desc="提取实体特征"):
        out_path = os.path.join(args.output_dir, f"{img_id}_entity.npz")

        # 断点续跑
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        # 图片不存在
        if not os.path.exists(img_path):
            print(f"[WARN] 图片不存在: {img_path}")
            errors += 1
            continue

        # 提取全局视觉特征
        img_feat = extract_global_image_feat(model, preprocess, img_path, args.device)
        if img_feat is None:
            errors += 1
            continue

        # 选 Top-M 实体
        sel_feats, sel_idx, sel_scores = select_top_m_entities(
            img_feat, ent_feats, args.top_m
        )

        # 记录实体名（方便 debug，不影响训练）
        sel_words = np.array([ent_words[i] for i in sel_idx], dtype=object)

        # 保存
        np.savez_compressed(
            out_path,
            feat    = sel_feats,      # (M, C)  —— 模型需要的
            indices = sel_idx,        # (M,)    —— debug 用
            scores  = sel_scores,     # (M,)    —— debug 用
            words   = sel_words,      # (M,)    —— debug 用
        )

    print("\n[Done]")
    print(f"  已处理: {len(items) - skipped - errors}")
    print(f"  跳过:   {skipped}")
    print(f"  错误:   {errors}")
    print(f"  输出目录: {args.output_dir}")

    # 简单验证最后一个文件
    if items:
        sample_id  = items[-1][0]
        sample_out = os.path.join(args.output_dir, f"{sample_id}_entity.npz")
        if os.path.exists(sample_out):
            sample = np.load(sample_out, allow_pickle=True)
            print(f"\n[验证] {sample_id}_entity.npz")
            print(f"  feat shape : {sample['feat'].shape}")   # (M, C)
            print(f"  Top-{args.top_m} 实体: {list(sample['words'])}")


if __name__ == "__main__":
    main()