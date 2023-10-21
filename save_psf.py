# load psf image and save the binary matrix to a npy file
import os
import numpy as np
import cv2
# psf_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/visualizations/66400_mask.png"
# save_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/visualizations/66400_mask.npy"
psf_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary/visualizations/171100_mask.png"
save_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary/visualizations/171100_mask.npy"
psf = cv2.imread(psf_file, cv2.IMREAD_GRAYSCALE)
psf = psf.astype(np.float32) / 255
psf = psf.astype(bool)
# transpose to [H, W]
psf = psf.transpose(1, 0)
# filp 180 degree
psf = np.flip(psf, axis=1)
np.save(save_file, psf)


