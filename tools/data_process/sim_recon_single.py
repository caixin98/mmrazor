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
def process_img(person_imgs_path):
 
    # dataset_path = '/mnt/workspace/RawSense/data/celebrity'
    # output_path = '/mnt/workspace/RawSense/data/celebrity_flatcam'
    output_path = "."
    noise_level = 10
    calib = loadmat('flatcam_calibdata.mat')  # load calibration data
    # flatcam.downsample_calib(calib)
    flatcam.clean_calib(calib)
    # Reconstruct
    lmbd = 3e-4  # L2 regularization parameter
    img_path = person_imgs_path
   
    if not os.path.exists(img_path) or not img_path.endswith('.png'):
        return
    img = cv2.imread(img_path)
    if img is None:
        print(img_path, "is not loaded")
        return
    # imaging process
    # img = cv2.resize(img, (256, 256))
    flat_img = flatcam.simulate_flatcam(img, calib)
    flat_img_rgb = flatcam.fc2bayer(flat_img, calib)
    flat_img_rgb = flatcam.bayer2rgb(flat_img_rgb)
    flat_img = flatcam.add_noise(flat_img, noise_level)
    recon = flatcam.fcrecon(flat_img, calib, lmbd)
    recon = recon * 255.
    recon = recon.astype(np.float32)
    # normalize the flatcam measurement
    print(flat_img_rgb.shape)
    # flat_img_rgb = flat_img_rgb - np.min(flat_img_rgb)
    # flat_img_rgb = flat_img_rgb / np.max(flat_img_rgb)
    flat_img_rgb = flat_img_rgb * 255.
    flat_img_rgb = flat_img_rgb.astype(np.float32)
    # recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
    # save the result
    file_name = os.path.basename(img_path)
    # recon_path = os.path.join(output_path, file_name)
    # os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    cv2.imwrite("test_flatcam.png", recon)
    cv2.imwrite("test_flatcam_rgb.png", flat_img_rgb)

    # break
    # cv2.imwrite(recon_path, recon)


if __name__ == "__main__":
    img_path = "test_img.png"
    process_img(img_path)
    