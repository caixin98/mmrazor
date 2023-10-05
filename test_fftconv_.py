from waveprop.pytorch_util import fftconvolve
import torch
import numpy as np
import cv2
# from scipy.signal import fftconvolve
A = [[1,1,1,1,1,0,0,0,0]]
# A = [0,0,1,1,1,1,1,0,0]
# A = [0,0,1,1,1]

B = [[1,1,1,1,1,1,1,1,1]]
A = torch.tensor(A).float()
B = torch.tensor(B).float()
C = fftconvolve(A, B)

# print(C[3:8])
print(C)
