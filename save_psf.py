# load psf image and save the binary matrix to a npy file
import os
import numpy as np
import cv2
# psf_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/visualizations/66400_mask.png"
# save_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/visualizations/66400_mask.npy"
# psf_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary/visualizations/171100_mask.png"
# save_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary/visualizations/171100_mask.npy"
#ali
# psf_file = "logs/distill/face/base_12800_no_align/visualizations/399300_psf.png"
# save_file = "logs/distill/face/base_12800_no_align/visualizations/399300_psf.npy"
#ali3
# psf_file = "logs/distill/face/base_no_psf_grad/visualizations/0_psf.png"
# save_file = "logs/distill/face/base_no_psf_grad/visualizations/0_psf.npy"
# psf_file = "logs/distill/face/base_12800/visualizations/413200_psf.png"
# save_file = "logs/distill/face/base_12800/visualizations/413200_psf.npy"

psf_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_shift_crop_size_240_expected_light_intensity=12800_binary/visualizations/99600_mask.png"
save_file = "logs/distill/face/vit2optical_bg_af_updater_rotate_shift_crop_size_240_expected_light_intensity=12800_binary/visualizations/99600_mask.npy"
psf = cv2.imread(psf_file, cv2.IMREAD_GRAYSCALE)
psf = psf.astype(np.float32) / 255
psf = psf.astype(bool)
# transpose to [H, W]
psf = psf.transpose(1, 0)
# filp 180 degree
psf = np.flip(psf, axis=1)
np.save(save_file, psf)


