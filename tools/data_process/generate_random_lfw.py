import os
import cv2
import numpy as np
import random
import os.path as osp
def pad_image_to_size(img, width, height, color=[0, 0, 0]):
    # Get the image dimensions
    h, w = img.shape[:2]

    # Calculate padding for each side
    pad_left = int(max(0, (width - w) / 2))
    pad_right = int(max(0, (width - w) - pad_left))
    pad_top = int(max(0, (height - h) / 2))
    pad_bottom = int(max(0, (height - h) - pad_top))

    # Apply the padding
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)

    return padded_img

class AddBackground(object):
    def __init__(self, img_dir, size = (160,160), prob=1.0, ratio=(0.8, 1.2), scale=(0.3, 2),):
        self.img_dir = img_dir
        self.prob = prob
        self.size = size
        self.ratio = ratio
        self.scale = scale
 
        self.background_files = os.listdir(img_dir)
        # load 3000 background images
        if len(self.background_files) > 1500:
            self.background_files = self.background_files[:1500]
        self.background_files = [cv2.imread(osp.join(img_dir, f)) for f in self.background_files]
    # RandomResizedCrop for images read from opencv
    def _transform(self, img):
        cropped_size = self.size
        ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        img_size = img.shape[:2]
        # random crop
        w = int(scale * cropped_size[0])
        h = int(scale * cropped_size[1])
        tw = int(w * ratio)
        th = int(h * ratio)
        i = np.random.randint(0, img_size[0] - th)
        j = np.random.randint(0, img_size[1] - tw)
        img = img[i:i + th, j:j + tw, :]
        # resize
        img = cv2.resize(img, cropped_size)
        return img
  
    def __call__(self, img):

        #random choose a background
        background = random.choice(self.background_files)
        background = self._transform(background)
        # padding img to size
        img = pad_image_to_size(img, self.size[0], self.size[1])
        img = np.where(img == 0, background, img)

        return img


def random_rotate_scale(img, angle, scale):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    transformed_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    return transformed_img

lfw_path = "/mnt/workspace/RawSense/data/lfw/lfw-112X96"
save_path = "/mnt/workspace/RawSense/data/lfw/lfw-112X96-random"
os.makedirs(save_path, exist_ok=True)
background_dir = "/mnt/workspace/RawSense/data/BG-20k/testval"
add_background = AddBackground(background_dir)
for person in os.listdir(lfw_path):
    person_path = os.path.join(lfw_path, person)
    save_person_path = os.path.join(save_path, person)
    os.makedirs(save_person_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(img_path, " is not loaded")
            continue
        
        angle = np.random.randint(-30, 30)
        scale = np.random.uniform(0.9, 1.1)
        
        transformed_img = random_rotate_scale(img, angle, scale)
        transformed_img = add_background(transformed_img) 
        save_img_path = os.path.join(save_person_path, img_name)
        cv2.imwrite(save_img_path, transformed_img)