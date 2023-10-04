# test affine 
import cv2
import torch
import numpy as np
import torch.nn.functional as F
#create a circle mask with radius 50 in a 500*500 image
img = np.zeros((500, 500), dtype=np.uint8)
cv2.circle(img, (250, 250), 50, 1, -1)
img = img.astype(np.float32)
cv2.imwrite("test_.jpg", img * 255)
img = torch.from_numpy(img)
img = img.unsqueeze(0).unsqueeze(0)
affine_para = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
affine_para = affine_para.unsqueeze(0)
grid = F.affine_grid(affine_para, img.size())
img = F.grid_sample(img, grid)
img = img.squeeze(0).squeeze(0).numpy()
img = (img - img.min()) / (img.max() - img.min())
img = (img * 255).astype(np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite("test.jpg", img)