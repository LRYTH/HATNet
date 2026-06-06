"""
Step 1: Build Entity Space
从现成的实体列表（txt 或 json）中读取名词实体，
用 GeoRSCLIP 文本编码器预计算所有实体的文本特征，
保存为 entity_space.npz 供后续步骤使用。

支持的实体列表格式：
  - txt：每行一个实体词，如 "aircraft\nbuilding\nroad\n..."
  - json：列表格式，如 ["aircraft", "building", "road", ...]
         或字典格式，如 {"entities": ["aircraft", ...]}
"""

import os
import json
import argparse
import numpy as np
import torch
import open_clip
from tqdm import tqdm

# ============================================================
# CONFIG —— 按你的实际路径修改
# ============================================================
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
CLIP_CKPT      = os.path.join(PROJECT_ROOT, "RS5M_ViT-H-14.pt")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 批量编码文本时的 batch size（防止显存溢出）
TEXT_BATCH_SIZE = 256
# ============================================================


def load_entity_list(entity_path: str) -> list[str]:
    """
    从 txt 或 json 文件中读取实体列表，
    去重、去空白、转小写后返回。
    """
    ext = os.path.splitext(entity_path)[-1].lower()

    if ext == ".txt":
        with open(entity_path, "r", encoding="utf-8") as f:
            entities = [line.strip().lower() for line in f if line.strip()]

    elif ext == ".json":
        with open(entity_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entities = [str(e).strip().lower() for e in data]
        elif isinstance(data, dict):
            # 尝试常见 key
            for key in ("entities", "words", "nouns", "vocab"):
                if key in data:
                    entities = [str(e).strip().lower() for e in data[key]]
                    break
            else:
                raise ValueError(
                    f"JSON 格式不识别，请确保 key 为 entities/words/nouns/vocab，"
                    f"实际 keys: {list(data.keys())}"
                )
        else:
            raise ValueError("JSON 内容既不是 list 也不是 dict")
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 .txt 或 .json")

    # 去重，保持顺序
    seen = set()
    unique_entities = []
    for e in entities:
        if e and e not in seen:
            seen.add(e)
            unique_entities.append(e)

    print(f"[Entity Space] 共加载 {len(unique_entities)} 个唯一实体")
    return unique_entities


def load_clip_model(ckpt_path: str, device: str):
    """加载 GeoRSCLIP (ViT-H-14) 并返回 (model, tokenizer)"""
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

    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    print("[CLIP] 模型加载完成")
    return model, tokenizer


@torch.no_grad()
def encode_entities(
    model,
    tokenizer,
    entities: list[str],
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    将实体列表编码为 CLIP 文本特征。
    prompt 模板：'An image contains {entity}.'（与论文一致）

    返回:
        feats: (E, C_clip)  float32 numpy array
    """
    prompts = [f"An image contains {e}." for e in entities]
    all_feats = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="编码实体文本"):
        batch_prompts = prompts[i : i + batch_size]
        tokens = tokenizer(batch_prompts).to(device)          # (B, L)
        feats  = model.encode_text(tokens).float()            # (B, C)
        # L2 归一化，方便后续余弦相似度计算
        feats  = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)                  # (E, C)


def main():
    parser = argparse.ArgumentParser(description="Build Entity Space for ECE")
    parser.add_argument(
        "--entity_path",
        type=str,
        required=True,
        help="实体列表文件路径（.txt 或 .json）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "entity_space.npz"),
        help="输出的 entity_space.npz 路径",
    )
    parser.add_argument(
        "--clip_ckpt",
        type=str,
        default=CLIP_CKPT,
        help="GeoRSCLIP 权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="cuda / cpu，例如 cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TEXT_BATCH_SIZE,
        help="文本编码批大小",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 1. 读取实体列表
    entities = load_entity_list(args.entity_path)

    # 2. 加载 CLIP
    model, tokenizer = load_clip_model(args.clip_ckpt, args.device)

    # 3. 编码
    feats = encode_entities(model, tokenizer, entities, args.device, args.batch_size)
    print(f"[Entity Space] 特征形状: {feats.shape}")   # (E, 1024) for ViT-H-14

    # 4. 保存
    #    entities_list 存为字符串数组，feat 存为 float32
    np.savez_compressed(
        args.output_path,
        feat=feats.astype(np.float32),                  # (E, C)
        entities=np.array(entities, dtype=object),       # (E,) str array
    )
    print(f"[Entity Space] 已保存至: {args.output_path}")
    print(f"  实体数量 E = {len(entities)}")
    print(f"  特征维度 C = {feats.shape[1]}")


if __name__ == "__main__":
    main()