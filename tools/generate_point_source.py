import cv2
import numpy as np
import os
output_root = '/root/caixin/RawSense/mmrazor/vis_optical/point_source'
#generate the point source image
def generate_point_source(img_size, point_pos, save_path):
    # img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    x = point_pos[0]
    y = point_pos[1]
    img[x-1:x+1, y-1:y+1, :] = 255

    cv2.imwrite(save_path, img)
    return img
img_size = [112,96]
#left-top, right-top, left-bottom, right-bottom and center

point_pos = [[1,1],[1,94],[110,1],[110,94],[55,47]]
for i in range(len(point_pos)):
    generate_point_source(img_size, point_pos[i], os.path.join(output_root, str(i)+'.png'))
