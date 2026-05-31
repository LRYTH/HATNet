"""
Extract GeoRSCLIP patch features for HATNet.
Output: (N_patches, 1280×3)
"""

import os
import json
import random

import numpy as np
from PIL import Image
import torch
import open_clip
from tqdm import tqdm
import torchvision.transforms as transforms


# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
DEVICE = "cuda:2"


# =========================
# Project root
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# =========================
# 固定全局随机种子
# =========================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =========================
# Model
# =========================
def create_model_and_preprocess():

    result = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained=os.path.join(PROJECT_ROOT, 'RS5M_ViT-H-14.pt')
    )

    # =========================
    # 兼容不同 open_clip 版本
    # =========================
    if isinstance(result, dict):
        model = result["model"]

    elif isinstance(result, tuple):

        if len(result) == 2:
            model, _ = result

        elif len(result) == 3:
            model, _, _ = result   # train / val split

        else:
            raise ValueError(f"Unexpected return format: {len(result)}")

    else:
        raise TypeError("Unknown open_clip return type")

    # =========================
    # 按论文要求的预处理 (论文 Section 2)
    # =========================
    # Image preprocessing: center crop to 224×224, horizontal flip, random rotation, normalization
    preprocess = transforms.Compose([
        transforms.CenterCrop(224),          # Center crop to 224×224
        transforms.RandomHorizontalFlip(),   # Horizontal flipping
        transforms.RandomRotation(degrees=15),  # Random rotation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    model = model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, preprocess


# =========================
# Hook: get patch tokens
# =========================
def forward_vit_multiscale_tokens(model, x):
    """
    Extract ViT patch tokens from layers 20, 25, 32.
    Returns:
        (B, N, 1280×3)
    """
    # patch embedding
    x = model.visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    cls = model.visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1],
                            device=x.device, dtype=x.dtype)
    x = torch.cat([cls, x], dim=1)
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    # 提取第 20、25、32 层的特征
    scale_features = []
    target_layers = [19, 24, 31]  # 0-indexed: 20th→19, 25th→24, 32nd→31

    for i, block in enumerate(model.visual.transformer.resblocks):
        x = block(x)
        if i in target_layers:
            feat = x.permute(1, 0, 2)  # (B, 1+N, 1280)
            feat = model.visual.ln_post(feat)
            feat = feat[:, 1:, :]  # 去掉 CLS token，只要 patch tokens
            scale_features.append(feat)  # (B, N, 1280)

    # 拼接三层特征
    feat = torch.cat(scale_features, dim=-1)  # (B, N, 3840)
    return feat


# =========================
# Feature extractor
# =========================
def extract_clip_features(model, preprocess, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = forward_vit_multiscale_tokens(model, image)
            # (1, N, 3840)

            feat = feat.squeeze(0)  # (N, 3840)

            # L2 归一化
            # feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

        return feat.cpu().numpy()

    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return None

#======================================RSICD和UCM数据集格式======================================
# =========================
# Main
# =========================
def main():

    SEED = 42  # ← 随机种子，可调
    set_seed(SEED)

    images_dir = os.path.join(PROJECT_ROOT, "datasets/UCM_captions/imgs")
    output_dir = os.path.join(PROJECT_ROOT, "data/UCM_GeoRSCLIP_cls_att")
    json_path = os.path.join(PROJECT_ROOT, "datasets/UCM_captions/dataset.json")

    # images_dir = os.path.join(PROJECT_ROOT, "datasets/RSICD/RSICD_images")
    # output_dir = os.path.join(PROJECT_ROOT, "data/RSICD_GeoRSCLIP_cls_att")
    # json_path = os.path.join(PROJECT_ROOT, "datasets/RSICD/annotations_rsicd/dataset_rsicd_modified.json")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model on {DEVICE}...")
    model, preprocess = create_model_and_preprocess()

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    images = dataset['images']

    # ← 加这一行，按 id 数字排序
    # images = sorted(images, key=lambda x: int(x.get('imgid', x.get('id', 0))))

    print(f"Total images: {len(images)}")

    for img_info in tqdm(images):

        filename = img_info.get('filename', '')
        img_id = img_info.get('imgid', img_info.get('id', ''))

        if not filename:
            continue

        img_path = os.path.join(images_dir, filename)

        if not os.path.exists(img_path):
            continue

        feat = extract_clip_features(model, preprocess, img_path)

        if feat is None:
            continue

        output_path = os.path.join(output_dir, f"{str(img_id)}.npz")

        np.savez_compressed(
            output_path,
            feat=feat.astype(np.float32)
        )

    print("Done.")
#======================================RSICD和UCM数据集格式======================================


#=========================================NWPU数据集格式=========================================
# def main():
#
#     SEED = 42  # ← 随机种子，可调
#     set_seed(SEED)
#
#     images_dir = os.path.join(PROJECT_ROOT, "datasets/NWPU-Captions/NWPU-RESISC45")
#     output_dir = os.path.join(PROJECT_ROOT, "data/NWPU_GeoRSCLIP_cls_att")
#     json_path  = os.path.join(PROJECT_ROOT, "datasets/NWPU-Captions/dataset_nwpu.json")
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     print(f"Loading model on {DEVICE}...")
#     model, preprocess = create_model_and_preprocess()
#
#     with open(json_path, 'r') as f:
#         dataset = json.load(f)
#
#     # ← NWPU 格式：按类别分组，需要展开
#     images = []
#     for category, img_list in dataset.items():
#         for img in img_list:
#             images.append({
#                 'imgid':    img['imgid'],
#                 'filename': img['filename'],
#                 'filepath': category,   # ← 类别名即子目录名
#             })
#
#     # 按 imgid 数字排序
#     # images = sorted(images, key=lambda x: int(x.get('imgid', 0)))
#
#     print(f"Total images: {len(images)}")
#
#     for img_info in tqdm(images):
#
#         filename = img_info.get('filename', '')
#         img_id   = img_info.get('imgid', '')
#         filepath = img_info.get('filepath', '')
#
#         if not filename:
#             continue
#
#         # ← 路径需要包含类别子目录
#         img_path = os.path.join(images_dir, filepath, filename)
#
#         if not os.path.exists(img_path):
#             print(f"[WARN] not found: {img_path}")
#             continue
#
#         feat = extract_clip_features(model, preprocess, img_path)
#
#         if feat is None:
#             continue
#
#         output_path = os.path.join(output_dir, f"{str(img_id)}.npz")
#
#         np.savez_compressed(
#             output_path,
#             feat=feat.astype(np.float32)
#         )
#
#     print("Done.")
#=========================================NWPU数据集格式=========================================

if __name__ == "__main__":
    main()