import json
data = json.load(open('datasets/UCM_captions/dataset.json'))
train_imgs = [img for img in data['images'] if img['split'] == 'train']
test_imgs = [img for img in data['images'] if img['split'] == 'test']

# 看看文件名是否有重复
train_names = set(img['filename'] for img in train_imgs)
test_names = set(img['filename'] for img in test_imgs)
print(f"重复文件: {train_names & test_names}")