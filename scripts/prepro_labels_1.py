"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

# 统计captions的单词，进行编码

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
from transformers import BertTokenizer

#######################################################################
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# vocab = tokenizer.get_vocab()
# # 将键值对中的键和值交换位置，并将键转换为字符串格式
# word_to_ix = {word: str(index) for word, index in vocab.items()}
# ix_to_word = {str(index): word for word, index in vocab.items()}
#######################################################################

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1 # 统计词频
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr] # 小于阈值的词
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words) # 小于阈值的词，出现的总共的次数
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = [] # 每一个图像都有一个 final_captions
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt] # 每一个txt, 词频大于阈值就是自己， 小于就编码为UNK
            img['final_captions'].append(caption)

    return vocab # 大于阈值的词，列表

def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length'] # 每个caption设定的最大长度
    N = len(imgs) # 图像数量
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i,img in enumerate(imgs):
        n = len(img['final_captions']) # 每个图像的有n个caption
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32') # 存储 一个图像的n个caption的编码

        for j,s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1

            for k,w in enumerate(s):
                if k < max_length:
                    Li[j,k] = wtoi[w] # 一个图像的第j个句子的第k个词 转换为编码

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0) # put all the labels together    所有caption的编码
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    # assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length #

#======================================RSICD和UCM数据集格式======================================
def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    # # ← 加这两行，按 id 数字大小排序
    # imgs = sorted(imgs, key=lambda x: x.get('imgid', x.get('id', 0)))

    seed(123) # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)

    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table      id to word
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table      word to id

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L) # 所有caption的编码，M个caption,
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length) # 所有caption的长度，M个长度
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need ，filepath和filename拼接
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        if params['images_root'] != '':
            # with Image.open(os.path.join(params['images_root'], img["category"], img['filename'])) as _img:
            with Image.open(os.path.join(params['images_root'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])
#======================================RSICD和UCM数据集格式======================================

#=========================================NWPU数据集格式=========================================
# def main(params):
#
#     raw_data = json.load(open(params['input_json'], 'r'))
#
#     # ← NWPU 格式：按类别分组，需要展开
#     imgs = []
#     for category, img_list in raw_data.items():
#         for img in img_list:
#             # 将 raw, raw_1, ..., raw_4 转换为 sentences 格式
#             captions = []
#             for key in ['raw', 'raw_1', 'raw_2', 'raw_3', 'raw_4']:
#                 if key in img and img[key].strip():
#                     tokens = img[key].strip().rstrip('.').lower().split()
#                     ## ← 打印出问题的图片
#                     # if len(tokens) == 0:
#                     #
#                     #     print(f"Empty tokens: imgid={img['imgid']}, key={key}, raw='{img[key]}'")
#                     # else:
#                     #     captions.append({'tokens': tokens})
#                     # ← 加这一行，过滤掉空的 tokens
#                     if len(tokens) > 0:
#                         captions.append({'tokens': tokens})
#
#                     # ← 加这一行，确保至少有一条标注
#             if len(captions) == 0:
#                 print(f"Warning: image {img['imgid']} has no valid captions, skipping")
#                 continue
#
#             imgs.append({
#                 'imgid':     img['imgid'],
#                 'split':     img['split'],
#                 'filename':  img['filename'],
#                 'filepath':  category,   # ← 类别名即子目录名
#                 'sentences': captions
#             })
#
#     # 按 imgid 数字排序
#     imgs = sorted(imgs, key=lambda x: int(x.get('imgid', 0)))
#
#     seed(123)  # make reproducible
#
#     # create the vocab
#     vocab = build_vocab(imgs, params)
#
#     itow = {i+1:w for i,w in enumerate(vocab)}
#     wtoi = {w:i+1 for i,w in enumerate(vocab)}
#
#     # encode captions in large arrays, ready to ship to hdf5 file
#     L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)
#
#     # create output h5 file
#     f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
#     f_lb.create_dataset("labels",        dtype='uint32', data=L)
#     f_lb.create_dataset("label_start_ix",dtype='uint32', data=label_start_ix)
#     f_lb.create_dataset("label_end_ix",  dtype='uint32', data=label_end_ix)
#     f_lb.create_dataset("label_length",  dtype='uint32', data=label_length)
#     f_lb.close()
#
#     # create output json file
#     out = {}
#     out['ix_to_word'] = itow
#     out['images'] = []
#     for i, img in enumerate(imgs):
#
#         jimg = {}
#         jimg['split']     = img['split']
#         jimg['file_path'] = os.path.join(img['filepath'], img['filename'])  # e.g. airplane/airplane_001.jpg
#         jimg['id']        = img['imgid']
#
#         if params['images_root'] != '':
#             img_full_path = os.path.join(
#                 params['images_root'],
#                 img['filepath'],   # ← 类别子目录
#                 img['filename']
#             )
#             with Image.open(img_full_path) as _img:
#                 jimg['width'], jimg['height'] = _img.size
#
#         out['images'].append(jimg)
#
#     json.dump(out, open(params['output_json'], 'w'))
#     print('wrote ', params['output_json'])
#=========================================NWPU数据集格式=========================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default="../datasets/RSICD/annotations_rsicd/dataset_rsicd_modified.json", help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='../data/RSICD/RSICD.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/RSICD/RSICD', help='output h5 file')
    parser.add_argument('--images_root', default='../datasets/RSICD/RSICD_images', help='root location in which images are stored, to be prepended to file_path in input json')

    # parser.add_argument('--input_json', default="../datasets/RSICD/annotations_rsicd/dataset_rsicd_m.json",
    #                     help='input json file to process into hdf5')
    # parser.add_argument('--output_json', default='../data/RSICD/RSICD.json', help='output json file')
    # parser.add_argument('--output_h5', default='../data/RSICD/RSICD', help='output h5 file')
    # parser.add_argument('--images_root', default='../datasets/RSICD/RSICD_images',
    #                     help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=50, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)

# 所有的 captions 都从头到尾 编码为一排， 可以通过第i个序号，获得第i个capton 的开始和结束位置

# {
#     "ix_to_word":{"0":"many"},
#     "images":[
#         {"split":"train", "file_path":"sdf/sdf.jpg","id":12, "width":224, "height":224} # 这里的id 就是 imgid
#     ]
# }

#17783 imgid = 2856
#55702