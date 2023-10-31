

import os
import shutil
import json

def restore_original_structure(path_dict_file, target_folder, restore_folder):
    with open(path_dict_file, 'r') as f:
        path_dict = json.load(f)
    file_list = os.listdir(target_folder)
    # the name is like MyPic_exp40000_index13231-23286490-0, key = 13231
    file_list.sort(key = lambda x: int(x.split('-')[0].split('index')[1]))
    for original_path, (current_path, unique_number) in path_dict.items():
        original_path = os.path.join(restore_folder, original_path)
        current_path = os.path.join(target_folder, file_list[unique_number])
        if not os.path.exists(original_path):
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            shutil.copy2(current_path, original_path)

if __name__ == '__main__':
    path_dict_file = 'path_dict.json'  # The file storing the path mapping
    target_folder = '/root/caixin/data/raw_lfw/test20231030'  # The folder where the images are currently located
    target_image_folder = '/root/caixin/data/raw_lfw/test20231030/Cam20231030'  # The folder where the images are currently located
    restore_folder = "/mnt/workspace/RawSense/data/raw_lfw/test20231030/restore"  # The folder to restore the original structure
    os.makedirs(restore_folder, exist_ok=True)
    path_dict_file = os.path.join(target_folder, path_dict_file)
    restore_original_structure(path_dict_file, target_image_folder, restore_folder)