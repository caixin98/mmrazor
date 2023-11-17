#create a subset for celeb dataset
import os
import cv2
import numpy as np
from tqdm import tqdm
import random
dataset_path = '/root/caixin/data/celebrity'
save_path = '/root/caixin/data/celebrity_subset_cropped'
label_root = '/root/caixin/data/celebrity/celebrity_label.txt'
imglist_root = '/root/caixin/data/celebrity/celebrity_data.txt'
os.makedirs(save_path, exist_ok=True)

img_list = []
with open(label_root) as f:
    label = [line.strip() for line in f.readlines()]
    labels = [int(l) for l in label]


with open(imglist_root) as f:
    data = [line.strip() for line in f.readlines()]
assert len(data) == len(labels)
subset_data = []
subset_label = []
subset_index = random.sample(range(len(data)), 100000)


# copy img to subset folder
# for i in subset_index:
#     subset_data.append(data[i])
#     subset_label.append(labels[i])
for i in tqdm(subset_index):
    data_ = data[i]
    img_path = os.path.join(dataset_path, data_)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(img_path, " is not loaded")
        continue
    # center crop the img to 260 * 223
    h, w, _ = img.shape
    if h < 223 or w < 260:
        print(img_path, " is too small")
        continue
    img = img[int((h - 223) / 2):int((h - 223) / 2) + 223, int((w - 260) / 2):int((w - 260) / 2) + 260, :]
    # save the result
    save_img_path = os.path.join(save_path, data_)
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    cv2.imwrite(save_img_path, img)
    subset_data.append(data_)
    subset_label.append(labels[i])

with open(os.path.join(save_path, 'celebrity_data.txt'), 'w') as f:
    f.write('\n'.join(subset_data))
with open(os.path.join(save_path, 'celebrity_label.txt'), 'w') as f:
    f.write('\n'.join([str(l) for l in subset_label]))