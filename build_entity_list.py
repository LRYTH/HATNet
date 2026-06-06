"""
从训练集 captions 中自动提取名词实体列表，保存为 entity_list.txt
支持数据集格式：
  - RSICD / UCM：{"images": [{"split": "train", "sentences": [{"raw": "..."}, ...]}]}
  - NWPU：       {category: [{"sentences": [{"raw": "..."}, ...]}]}

用法：
  python build_entity_list.py \
      --json_path  datasets/UCM_captions/dataset.json \
      --output     data/entity_list.txt \
      --split      train \
      --min_freq   2
"""

import os
import re
import json
import argparse
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# 保证所需资源已下载
for _pkg in ['averaged_perceptron_tagger_eng', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(_pkg, quiet=True)

# ──────────────────────────────────────────────────────────
# 过滤词表：这些词即使是名词也无语义价值
# ──────────────────────────────────────────────────────────
STOPWORDS = {
    'image', 'photo', 'picture', 'view', 'scene', 'area', 'region',
    'lot', 'number', 'kind', 'type', 'way', 'part', 'side', 'top',
    'bottom', 'left', 'right', 'center', 'middle', 'background',
    'foreground', 'line', 'piece', 'set', 'group', 'row', 'column',
    'shape', 'color', 'colour', 'size', 'land', 'ground', 'place',
    'spot', 'thing', 'object', 'item', 'area', 'space', 'surface',
}

lemmatizer = WordNetLemmatizer()


def clean_caption(text: str) -> str:
    """基础清洗：转小写，去除标点"""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_nouns_from_caption(caption: str) -> list[str]:
    """
    用 NLTK POS tagger 提取名词（NN / NNS / NNP / NNPS），
    还原词形（lemmatize），过滤停用词和单字符词。
    """
    tokens = word_tokenize(clean_caption(caption))
    tagged = pos_tag(tokens)

    nouns = []
    for word, tag in tagged:
        if tag.startswith('NN'):                       # 所有名词词性
            lemma = lemmatizer.lemmatize(word, 'n')    # 复数 → 单数
            if (
                len(lemma) > 1                         # 去除单字符
                and lemma not in STOPWORDS             # 去除无意义词
                and lemma.isalpha()                    # 只保留纯字母
            ):
                nouns.append(lemma)
    return nouns


def iter_captions(json_path: str, split: str) -> list[str]:
    """
    从 json 文件中迭代指定 split 的所有 caption 文本。
    自动识别 RSICD/UCM 和 NWPU 两种格式。
    split='all' 时不过滤，提取全部 caption。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    captions = []

    if isinstance(data, dict) and 'images' in data:
        # ── RSICD / UCM 格式 ─────────────────────────────────────
        for img in data['images']:
            img_split = img.get('split', 'train')
            if split != 'all' and img_split != split:
                continue
            for sent in img.get('sentences', []):
                raw = sent.get('raw', sent.get('tokens', ''))
                if isinstance(raw, list):
                    raw = ' '.join(raw)
                if raw:
                    captions.append(raw)

    elif isinstance(data, dict):
        # ── NWPU 格式：{category: [{sentences:[{raw:...}]}]} ─────
        # NWPU 没有 split 字段，默认全取（split 参数忽略）
        for category, img_list in data.items():
            for img in img_list:
                for sent in img.get('sentences', []):
                    raw = sent.get('raw', '')
                    if raw:
                        captions.append(raw)
    else:
        raise ValueError(f"不支持的 JSON 格式，根类型: {type(data)}")

    return captions


def build_entity_list(
    json_path: str,
    output_path: str,
    split: str = 'train',
    min_freq: int = 2,
) -> list[str]:
    """
    主流程：
      1. 读取指定 split 的所有 caption
      2. 用 NLTK 提取名词并统计词频
      3. 过滤低频词（< min_freq）
      4. 按词频降序保存到 output_path
    """
    print(f"[Step 1] 读取 captions：{json_path}  split={split}")
    captions = iter_captions(json_path, split)
    print(f"         共 {len(captions)} 条 caption")

    print("[Step 2] 提取名词实体（NLTK POS tagging）...")
    counter = Counter()
    for cap in captions:
        nouns = extract_nouns_from_caption(cap)
        counter.update(nouns)

    print(f"         原始名词种类: {len(counter)}")

    # 过滤低频
    filtered = {w: c for w, c in counter.items() if c >= min_freq}
    print(f"         过滤后（>= {min_freq} 次）: {len(filtered)} 个实体")

    # 按词频降序排列
    entities = sorted(filtered.keys(), key=lambda w: -filtered[w])

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in entities:
            f.write(word + '\n')

    print(f"\n[Done] 实体列表已保存至: {output_path}")
    print(f"       共 {len(entities)} 个实体")
    print(f"\n词频 Top-20 预览:")
    for w in entities[:20]:
        print(f"  {w:<20} {filtered[w]} 次")

    return entities


def main():
    parser = argparse.ArgumentParser(description="从 captions 中提取名词实体列表")
    parser.add_argument('--json_path',  type=str, required=True,
                        help="数据集 JSON 路径（dataset.json 等）")
    parser.add_argument('--output',     type=str, default='data/entity_list.txt',
                        help="输出的实体列表 txt 路径")
    parser.add_argument('--split',      type=str, default='train',
                        choices=['train', 'val', 'test', 'all'],
                        help="只用哪个 split 的 captions（默认 train）")
    parser.add_argument('--min_freq',   type=int, default=2,
                        help="最低词频，低于此值的词过滤掉（默认 2）")
    args = parser.parse_args()

    build_entity_list(
        json_path   = args.json_path,
        output_path = args.output,
        split       = args.split,
        min_freq    = args.min_freq,
    )


if __name__ == '__main__':
    main()