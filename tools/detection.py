from optical_conv import optical_conv
from mmcv import Config
from mmcls.datasets.pipelines import Compose

cfg = Config.fromfile('configs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix.py')
cfg.data.val.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='Propagated',
                keys=['img'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.54,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 30),
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=0.0,
                    return_translate=True,
                ),
            dict(type='Affine2label',),
            # dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/testval',size = (100, 100),is_tensor=True),
     
            # dict(type='Collect', keys=['img', 'affine_matrix'],meta_keys=['image_file','affine_matrix'])
            dict(type='Collect', keys=['img', 'affine_matrix','target','target_weight'],meta_keys=['image_file'])
]

#generate a picture with muti faces, the faces are not overlapped with each other
def generate_muti_faces(faces_path):
    import os
    import cv2
    import numpy as np
    faces = []
    for face in faces_path:
        # if not face.endswith(".png") or face.endswith(".jpg"):
        #     continue
        print(face)
        face = cv2.imread(face)
        faces.append(face)
    
    # faces_img = np.zeros((112, 96*len(faces), 3))
    # the left-top has one face 
    # the right-bottom has one face
    faces_img = np.zeros((112*len(faces), 96*len(faces), 3))
    for i in range(len(faces)):
        faces_img[i*112:(i+1)*112, i*96:(i+1)*96, :] = faces[i]
    
    cv2.imwrite("vis_dection/muti_faces.png", faces_img)
    return faces


# pipeline = Compose(cfg.data.val.pipeline)
cfg.algorithm.architecture.model.backbone.optical.load_weight_path = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/latest.pth"
# source_img_path = "/root/caixin/data/lfw/lfw-112X96/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
# face_path = ["/root/caixin/data/lfw/lfw-112X96/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "/root/caixin/data/lfw/lfw-112X96/Alan_Ball/Alan_Ball_0002.jpg"]
# muti_face = generate_muti_faces(face_path)
muti_face_path = "vis_dection/muti_faces.png"

optical_conv(muti_face_path, cfg, target_dir = "vis_dection")

muti_face_after_optical_path = "vis_dection/muti_faces_after_optical.png"

cfg.algorithm.architecture.model.backbone.optical = dict(
    type='LoadPsf',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[240, 200],
    center_crop_size=[240, 200],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
    expected_light_intensity=12800,
    do_affine = True,
    requires_grad_psf = False,
    # binary=True,
    load_psf_path = "/root/caixin/RawSense/mmrazor/vis_dection/Aaron_Eckhart_0001_after_optical.png",
    n_psf_mask=1)

cfg.data.val.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='Propagated',
                keys=['img'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.54,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 30),
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=0.0,
                    return_translate=True,
                ),
            dict(type='Affine2label',),
            # dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/testval',size = (100, 100),is_tensor=True),
     
            # dict(type='Collect', keys=['img', 'affine_matrix'],meta_keys=['image_file','affine_matrix'])
            dict(type='StackImagePair', keys=['img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img', 'affine_matrix','target','target_weight'],meta_keys=['image_file'])
]


# optical_conv(muti_face_after_optical_path, cfg, target_dir = "vis_dection")
# conduct the conv with a kernel image and a source image
source_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/muti_faces_before_optical.png"
kernel_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/Aaron_Eckhart_0001_before_optical.png"
#implent a conv algorithm in the following function

import cv2
import numpy as np

def read_and_convert_to_grayscale(image_path):
    """ Read the image from the path and convert it to grayscale """
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_kernel(kernel):
    """ Normalize the kernel so that all of the values sum up to 1 """
    return kernel / np.sum(kernel)

def conv(kernel_img_path, source_img_path, output_path = "vis_dection/conv_result.png"):
    # Read the images
    kernel = read_and_convert_to_grayscale(kernel_img_path)
    source = read_and_convert_to_grayscale(source_img_path)
    # print(kernel.shape, source.shape)
    # print(kernel_img_path, kernel)
    # Normalize the kernel
    kernel = normalize_kernel(kernel)

    # Perform the convolution
    result = cv2.filter2D(source, -1, kernel)
    # normalize the result
    result = result / np.max(result) * 255
    result = result.astype(np.uint8)
    # print("result", result)
    # Save the result
    cv2.imwrite(output_path, result)
    return result

result = conv(kernel_img_path, source_img_path)

source_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/muti_faces_after_optical_after_optical.png"
kernel_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/Aaron_Eckhart_0001_after_affine.png"
img = cv2.imread(source_img_path, 0)
template = cv2.imread(kernel_img_path, 0)
res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
# res = res / np.max(res) * 255
# threshold = 0.8
# loc = np.where(res >= threshold)
# w, h = template.shape[::-1]
# # 用方框标记出匹配区域
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imwrite("vis_dection/matchTemplate.png", img)

# muti_face_source_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/muti_faces_before_optical.png"
# single_face_source_img_path = "/root/caixin/RawSense/mmrazor/vis_dection/Aaron_Eckhart_0001_before_optical.png"
# psf_path = "/root/caixin/RawSense/mmrazor/logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/visualizations/220500_psf.png"
# single_conv_result = conv(psf_path, source_img_path, output_path = "vis_dection/single_conv_result.png")
# muti_conv_result = conv(psf_path, muti_face_source_img_path, output_path = "vis_dection/muti_conv_result.png")
# single_muti_conv_result = conv("vis_dection/single_conv_result.png", "vis_dection/muti_conv_result.png", output_path = "vis_dection/single_muti_conv_result.png")