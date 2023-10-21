from waveprop.pytorch_util import fftconvolve
import torch
import numpy as np
# import cv2
# from scipy.signal import fftconvolve
# A = [1,1,1,1,0,0,0,0,0]
# # A = [0,0,0,0,1,2,3,4,5]
# # A = [0,0,1,1,1]

# B = [1,1,1,1,1,1,1,1,1]
# # A = torch.tensor(A).float()
# # B = torch.tensor(B).float()
# C = fftconvolve(A, B, mode='same')
# # C = fftconvolve(A, B)

# # print(C[3:8])
# print(C)
# test affine 
import cv2
import torch
import numpy as np
import torch.nn.functional as F
#create a circle mask with radius 50 in a 500*500 image
img = np.zeros((500, 500), dtype=np.uint8)
cv2.circle(img, (250, 250), 50, 1, -1)
img = img.astype(np.float32)
kernel = np.zeros((500, 500), dtype=np.uint8)
#draw a X in the kernel
cv2.line(kernel, (0, 0), (500, 500), 1, 5)
cv2.line(kernel, (0, 500), (500, 0), 1, 5)
img = torch.from_numpy(img)
img = img.unsqueeze(0).unsqueeze(0)
kernel = torch.from_numpy(kernel)
kernel = kernel.unsqueeze(0).unsqueeze(0)
after_conv = fftconvolve(img, kernel, mode='same')
affine_para = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
affine_para = affine_para.unsqueeze(0)
grid = F.affine_grid(affine_para, img.size())
img = F.grid_sample(img, grid)
img = img.squeeze(0).squeeze(0).numpy()
img = (img - img.min()) / (img.max() - img.min())
img = (img * 255).astype(np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite("test.jpg", img)