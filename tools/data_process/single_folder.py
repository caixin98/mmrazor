import os
import shutil
import json

def recursive_image_crawl(root_folder, target_folder, path_dict, unique_number):
    root_folder = root_folder + "/" if not root_folder.endswith("/") else root_folder
    target_folder = target_folder + "/" if not target_folder.endswith("/") else target_folder
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                source = os.path.join(dirpath, filename)
                new_filename = f"{unique_number}_{filename}"
                target = os.path.join(target_folder, new_filename)

                shutil.copy2(source, target)
                path_dict[source.replace(root_folder,"")] = (target.replace(target_folder,""), unique_number)
                unique_number += 1

    return path_dict, unique_number


if __name__ == '__main__':
    # root_folder = "/mnt/workspace/RawSense/data/celebrity_subset"
    # target_folder = '/mnt/workspace/RawSense/data/celebrity_subset_single'
    root_folder = "/mnt/workspace/RawSense/data/lfw/lfw-112X96-random"
    target_folder = '/mnt/workspace/RawSense/data/lfw-112X96-random-single'
    unique_number = 0
    path_dict = dict()
    os.makedirs(target_folder, exist_ok=True)

    path_dict, unique_number = recursive_image_crawl(root_folder, target_folder, path_dict, unique_number)

    with open(os.path.join(target_folder, 'path_dict.json'), 'w') as f:
        json.dump(path_dict, f)