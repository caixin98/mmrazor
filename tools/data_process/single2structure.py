

import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

def restore_original_structure(path_dict_file, target_folder, restore_folder):
    with open(path_dict_file, 'r') as f:
        path_dict = json.load(f)
    file_list = os.listdir(target_folder)
    #filter all file not start with MyPic
    file_list = [x for x in file_list if x.startswith('MyPic')]
    # the name is like 
    # MyPicFast_exp18000_index0-23286490-0
    # MyPic_exp40000_index13231-23286490-0, 
    file_list.sort(key = lambda x: int(x.split('-')[0].split('index')[1]))
    for original_path, (current_path, unique_number) in path_dict.items():
        original_path = os.path.join(restore_folder, original_path)
        current_path = os.path.join(target_folder, file_list[unique_number])
        if not os.path.exists(original_path):
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            shutil.copy2(current_path, original_path)

if __name__ == '__main__':
    dataset_id = 'Cam20231117-1'
    path_dict_file = 'lfw_path_dict.json'  # The file storing the path mapping
    target_folder = '/root/caixin/data/%s'%dataset_id  # The folder where the images are currently located
    target_image_folder = '/root/caixin/data/%s'%dataset_id  # The folder where the images are currently located
    restore_folder = "/mnt/workspace/RawSense/data/raw_lfw/%s/restore"%dataset_id  # The folder to restore the original structure
    os.makedirs(restore_folder, exist_ok=True)
    # path_dict_file = os.path.join(target_folder, path_dict_file)
    restore_original_structure(path_dict_file, target_image_folder, restore_folder)

    dataset_path = '/root/caixin/data/raw_lfw/%s/restore'%dataset_id
    save_path = '/root/caixin/data/raw_lfw/%s/prosessed'%dataset_id
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
            save_img_path = os.path.join(save_path, person, img_name)
            if os.path.exists(save_img_path):
                continue
            # imaging process
            img = cv2.resize(img, (308, 257))
            img = cv2.flip(img, 0)
            img = np.rot90(img)
            # save the result
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            # print(save_img_path)
            cv2.imwrite(save_img_path, img)