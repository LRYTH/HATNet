"""
Create a reference json file for validation/test evaluation with coco-caption repo.
For NWPU dataset format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse


def main(params):
    raw_data = json.load(open(params['input_json'], 'r'))

    # Create output json file in COCO format
    # {"images": [...], "annotations": [...]}

    out = {}
    out.update({'images': [], 'annotations': []})

    cnt = 0
    for category, img_list in raw_data.items():
        for img in img_list:
            # 只处理 val 和 test split
            if img['split'] not in ['val', 'test']:
                continue

            # 添加图片信息
            out['images'].append({'id': img['imgid']})

            # 添加所有 5 条标注
            for key in ['raw', 'raw_1', 'raw_2', 'raw_3', 'raw_4']:
                if key in img and img[key].strip():
                    caption = img[key].strip()
                    out['annotations'].append({
                        'image_id': img['imgid'],
                        'caption': caption,
                        'id': cnt
                    })
                    cnt += 1

    json.dump(out, open(params['output_json'], 'w'))
    print(f'wrote {params["output_json"]}')
    print(f'Total images: {len(out["images"])}')
    print(f'Total annotations: {len(out["annotations"])}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json',
                        default="/home/shenxiang/Zhoumiao/HATNet/datasets/NWPU-Captions/dataset_nwpu.json",
                        help='input json file (NWPU format)')
    parser.add_argument('--output_json',
                        default='/home/shenxiang/Zhoumiao/HATNet/data/NWPU/NWPU_val.json',
                        help='output json file (COCO format for evaluation)')

    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)