# simulate the imaging and reconstruction process
# transform the image to the flatcam domain and reconstruct it back
# dataset_path = '/mnt/workspace/RawSense/data/celebrity'
# output_path = '/mnt/workspace/RawSense/data/celebrity_flatcam'
# import os
# import numpy as np
# import cv2
# import flatcam
# from tqdm import tqdm
# import flatcam
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# # Load data
# meas = mpimg.imread('sample_capture.png')  # load flatcam measurement
# calib = loadmat('flatcam_calibdata.mat')  # load calibration data
# flatcam.clean_calib(calib)
# # Reconstruct
# lmbd = 3e-4  # L2 regularization parameter

# for person_imgs in tqdm(os.listdir(dataset_path)):
#     imgs_path = os.path.join(dataset_path, person_imgs)
#     if not os.path.isdir(imgs_path):
#         continue
#     for img in os.listdir(imgs_path):
#         file_name = os.path.join(person_imgs, img)
#         img_path = os.path.join(dataset_path, file_name)
#         if not os.path.exists(img_path):
#             continue
#         if not img.endswith('.jpg'):
#             continue
#         img = cv2.imread(img_path)
#         if img is None:
#             print(img_path, " is not loaded")
#             continue
#         # imaging process
#         img = cv2.resize(img, (256, 256))
#         flat_img = flatcam.simulate_flatcam(img, calib)
#         recon = flatcam.fcrecon(flat_img, calib, lmbd)
#         recon = recon * 255.
#         recon = recon.astype(np.float32)
#         # recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
#         # save the result
#         # flat_img_path = os.path.join(output_path, file_name)
#         recon_path = os.path.join(output_path, file_name)
#         # os.makedirs(os.path.dirname(flat_img_path), exist_ok=True)
#         os.makedirs(os.path.dirname(recon_path), exist_ok=True)
#         # cv2.imwrite(flat_img_path, flat_img)
#         cv2.imwrite(recon_path, recon)
#         # break

import os
import cv2
import numpy as np
import flatcam
import multiprocessing
from tqdm import tqdm
from scipy.io import loadmat
import time

def process_img(person_imgs):
    # dataset_path = '/mnt/workspace/RawSense/data/celebrity'
    # output_path = '/mnt/workspace/RawSense/data/celebrity_flatcam'
    dataset_path = '/mnt/workspace/RawSense/data/lfw/lfw-112X96'
    output_path = '/mnt/workspace/RawSense/data/lfw/lfw-flatcam'
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    # flatcam.downsample_calib(calib)
    flatcam.clean_calib(calib)
    # Reconstruct
    lmbd = 3e-4  # L2 regularization parameter
    imgs_path = os.path.join(dataset_path, person_imgs)
    if not os.path.isdir(imgs_path):
        return
    for img_name in os.listdir(imgs_path):
        file_name = os.path.join(person_imgs, img_name)
        img_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(img_path) or not img_path.endswith('.jpg'):
            return
        img = cv2.imread(img_path)
        if img is None:
            print(img_path, "is not loaded")
            return
        # imaging process
        img = cv2.resize(img, (256, 256))
        start = time.time()
        flat_img = flatcam.simulate_flatcam(img, calib)
        recon = flatcam.fcrecon(flat_img, calib, lmbd)
        print(time.time() - start)
        recon = recon * 255.
        recon = recon.astype(np.float32)
        # recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        # save the result
        recon_path = os.path.join(output_path, file_name)
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)
        cv2.imwrite("test.png", recon)
        break
        cv2.imwrite(recon_path, recon)

# def list_files_in_directory(directory_path):
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             yield os.path.join(root, file)

# dataset_path = '/mnt/workspace/RawSense/data/celebrity'
dataset_path = '/mnt/workspace/RawSense/data/lfw/lfw-112X96'

# all_image_files = list_files_in_directory(dataset_path)
# calculate the number of images
person_imgs = os.listdir(dataset_path)
person_imgs.sort()
num_folder = len(person_imgs)  >> 1
# person_imgs = person_imgs[:num_folder]
# person_imgs = person_imgs[:num_folder]

# def count_files_in_directory(directory):
#     return len([f for dirpath, dirnames, files in os.walk(directory) for f in files])
# num_images = count_files_in_directory(dataset_path)
# print("total number of images: ", num_images)

# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 4))
#     for _ in tqdm(pool.imap_unordered(process_img, person_imgs), total=len(person_imgs)):
#         pass
if __name__ == "__main__":
    for person in tqdm(person_imgs):
        process_img(person)
        break