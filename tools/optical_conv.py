from mmcls.models import build_optical
from mmcv import Config
from mmcls.datasets.pipelines import Compose
import os
from torchvision.utils import save_image
cfg = Config.fromfile('configs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix.py')
pipeline = Compose(cfg.data.val.pipeline)
# cfg.algorithm.architecture.model.backbone.optical.load_weight_path = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/latest.pth"
cfg.algorithm.architecture.model.backbone.optical.expected_light_intensity = 1

def optical_conv(source_img_path,cfg):
    source_filename = os.path.basename(source_img_path)
    source_dir = os.path.dirname(source_img_path)
    target_dir = "vis_optical"
    optical = build_optical(cfg.algorithm.architecture.model.backbone.optical)
    source_dict = dict(img_info=dict(filename=source_filename))
    source_dict['img_prefix'] = source_dir
    source = pipeline(source_dict)
    # print(source["img"].device, optical._psf.device)
    optical(source["img"].to(optical._psf.device))
    # visualize the result 

    after_optical = optical.after_optical
    before_optical = optical.before_optical
    after_affine = optical.after_affine
    # create save path
    os.makedirs(target_dir, exist_ok=True)
    # save the result
    after_optical = save_image(after_optical,
                               os.path.join(target_dir, '%s_after_optical.png'%source_filename.split('.')[0]),
                               normalize = True)
    before_optical = save_image(before_optical,os.path.join(target_dir, '%s_before_optical.png'%source_filename.split('.')[0]), normalize = True)
    after_affine = save_image(after_affine, os.path.join(target_dir, '%s_after_affine.png'%source_filename.split('.')[0]), normalize = True)

if __name__ == "__main__":
    
    # source_imgs_path = "/root/caixin/RawSense/mmrazor/vis_optical/point_source"
    source_imgs_path = "/mnt/data/oss_beijing/caixin/teaser2"


    for point in os.listdir(source_imgs_path):
        if not point.endswith(".png") or point.endswith(".png"):
            continue
        source_img_path = os.path.join(source_imgs_path, point)
    # source_img_path = "/root/caixin/data/lfw/lfw-112X96/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
        optical_conv(source_img_path,cfg)

  
