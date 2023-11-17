#rotate 90 degree counterclockwise and flip horizontally
import os
import cv2
import numpy as np
from tqdm import tqdm  
dataset_path = '/root/caixin/data/raw_lfw/zimage20231112/restore'
save_path = '/root/caixin/data/raw_lfw/zimage20231112/prosessed'
os.makedirs(save_path, exist_ok=True)
for person in tqdm(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        if not os.path.exists(img_path):
            continue
        if not img_name.endswith('.jpg'):
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(img_path, " is not loaded")
            continue
        # imaging process
        img = cv2.resize(img, (308, 257))
        img = cv2.flip(img, 0)
        img = np.rot90(img)
        # save the result
        save_img_path = os.path.join(save_path, person, img_name)
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        # print(save_img_path)
        cv2.imwrite(save_img_path, img)
    #     break
    # break