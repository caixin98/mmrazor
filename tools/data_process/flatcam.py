import numpy as np
import math
from scipy.ndimage import rotate as imrotate
from numpy.linalg import multi_dot

import numpy as np
from skimage.transform import resize
from skimage import img_as_float
import time
import cv2
def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['cSize'] = np.squeeze(calib['cSize'])
    calib['angle'] = np.squeeze(calib['angle'])


def obtain_calib_svd( calib ):
    calib_svd = calib # note, what happens to calib_svd happens to calib. To make just a copy, calib_svd = dict(calib)
    clean_calib(calib_svd)
    P1 = np.dstack((calib['P1r'], calib['P1gb'], calib['P1gr'], calib['P1b']))
    print(P1.shape)
    Q1 = np.dstack((calib['Q1r'], calib['Q1gb'], calib['Q1gr'], calib['Q1b']))
    # Initialize new entries for calib data struct
    calib_svd['UL_all'] = np.empty([P1.shape[0], P1.shape[0], 4])
    calib_svd['DL_all'] = np.empty([P1.shape[0], P1.shape[1], 4])
    calib_svd['VL_all'] = np.empty([P1.shape[1], P1.shape[1], 4])
    calib_svd['singL_all'] = np.empty([P1.shape[1], 4])
    calib_svd['UR_all'] = np.empty([Q1.shape[0], Q1.shape[0], 4])
    calib_svd['DR_all'] = np.empty([Q1.shape[0], Q1.shape[1], 4])
    calib_svd['VR_all'] = np.empty([Q1.shape[1], Q1.shape[1], 4])
    calib_svd['singR_all'] = np.empty([Q1.shape[1], 4])
    for i in range(4):
        # Left matrices (P1)
        u, s, vh = np.linalg.svd(P1[:, :, i], full_matrices=True)
        calib_svd['UL_all'][:, :, i] = u
        calib_svd['DL_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([P1.shape[0] - s.size, s.size])))
        calib_svd['VL_all'][:, :, i] = vh.T
        calib_svd['singL_all'][:, i] = s
        # Right matrices (Q1)
        u, s, vh = np.linalg.svd(Q1[:, :, i], full_matrices=True)
        calib_svd['UR_all'][:, :, i] = u
        calib_svd['DR_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([Q1.shape[0] - s.size, s.size])))
        calib_svd['VR_all'][:, :, i] = vh.T
        calib_svd['singR_all'][:, i] = s


def fc2bayer( im, calib ):
    # split up different color channels
    r = im[1::2, 1::2]
    gb = im[0::2, 1::2]
    gr = im[1::2, 0::2]
    b = im[0::2, 0::2]
    Y = np.dstack([r, gb, gr, b])
    # rotate capture
    M = cv2.getRotationMatrix2D((Y.shape[1]/2, Y.shape[0]/2), -calib['angle'], 1)
    rotate_shape = Y.shape
    cv2.warpAffine(Y, M, (rotate_shape[1], rotate_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # Y = imrotate(Y, calib['angle'], reshape=False)
    # crop usable sensor measurements
    csize = calib['cSize']
    start_row = int((Y.shape[0] - csize[0])/2)
    end_row = int(start_row + csize[0]) # omit -1 because Python indexing does not include end index
    start_col = int((Y.shape[1] - csize[1])/2)
    end_col = int(start_col + csize[1])
    Y = Y[start_row:end_row, start_col:end_col, :]
    return Y


def make_separable( Y ):
    # start = time.time()
    # print("Y.mean", time.time() - start)
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def bayer2rgb( X_bayer, normalize=True ):

    # X_rgb = np.empty(X_bayer.shape[:-1] + (3,))  # here, + means append to the tuple
    X_rgb = X_bayer[:, :, :3]
    X_rgb[:, :, 0] = X_bayer[:, :, 0]
    X_rgb[:, :, 1] = 0.5 * (X_bayer[:, :, 1] + X_bayer[:, :, 2])
    X_rgb[:, :, 2] = X_bayer[:, :, 3]

    # normalize to be from 0 to 1
    if normalize:

        X_rgb = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min())
    
    return X_rgb


def fcrecon( cap, calib, lmbd):
    # check if SVDs have been taken

    if not 'UL_all' in calib:
        obtain_calib_svd(calib)
    
    Y = fc2bayer(cap, calib)  # convert RAW output to Bayer color channels
    start = time.time()

    Y = make_separable(Y) # let rows and columns have 0-mean
    # print("make_separable", time.time() - start)

    X_bayer = np.empty([calib['VL_all'].shape[0], calib['VR_all'].shape[0], 4])

    start = time.time()
    for c in range(4):
        UL = calib['UL_all'][:, :, c]
        DL = calib['DL_all'][:, :, c]
        VL = calib['VL_all'][:, :, c]
        singLsq = np.square(calib['singL_all'][:, c])
        UR = calib['UR_all'][:, :, c]
        DR = calib['DR_all'][:, :, c]
        VR = calib['VR_all'][:, :, c]
        singRsq = np.square(calib['singR_all'][:, c])
        Yc = Y[:, :, c]
        inner = multi_dot([DL.T,UL.T,Yc,UR,DR]) / (np.outer(singLsq, singRsq) + np.full(X_bayer.shape[0:1], lmbd))
        X_bayer[:, :, c] = multi_dot([VL, inner, VR.T])
    # print("fcrecon", time.time() - start)
    
    X_bayer = X_bayer.clip(min=0)  # non-negative constraint: set all negative values to 0
    return bayer2rgb(X_bayer, True)  # bring back to RGB and normalize

def simulate_flatcam(input_im, calib):
    fc_dim = 256
    start = time.time()

    # Simulation
    if len(input_im.shape) == 2: # turn grayscale into rgb format
        input_im = np.stack((input_im,)*3, axis=-1)
    
    # Resize by forcing larger dimension to be 256
    if input_im.shape[0] > input_im.shape[1]:
        resize_dim = (256, int(input_im.shape[1] * 256 / input_im.shape[0]))
    else:
        resize_dim = (int(input_im.shape[0] * 256 / input_im.shape[1]), 256)
    

    input_im = img_as_float(resize(input_im, resize_dim, mode='reflect', anti_aliasing=True))

    # Pad and remember start and endpoints of actual image
    start_y = int((fc_dim - input_im.shape[0]) / 2)
    end_y = start_y + input_im.shape[0]
    start_x = int((fc_dim - input_im.shape[1]) / 2)
    end_x = start_x + input_im.shape[1]
    orig_im = np.zeros((fc_dim, fc_dim, 3))
    orig_im[start_y:end_y, start_x:end_x, :] = input_im

    # Simulate FlatCam capture, rotate measurements
    sim_fc = np.zeros((calib['cSize'][0] * 2, calib['cSize'][1] * 2))
    # print(orig_im[:, :, 0].shape)
    # print(sim_fc[1::2, 1::2].shape)
    # print(sim_fc[::2, 1::2].shape)

    rotate_shape = sim_fc[1::2, 1::2].shape
    M = cv2.getRotationMatrix2D((rotate_shape[1]/2, rotate_shape[0]/2), float(calib['angle']), 1)
    # print("calib['angle']", calib['angle'])
    sim_fc[1::2, 1::2] = cv2.warpAffine(np.dot(np.dot(calib['P1r'], orig_im[:, :, 0]), calib['Q1r'].T), M, (rotate_shape[1], rotate_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    sim_fc[::2, 1::2] = cv2.warpAffine(np.dot(np.dot(calib['P1gb'], orig_im[:, :, 1]), calib['Q1gb'].T), M, (rotate_shape[1], rotate_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    sim_fc[1::2, ::2] = cv2.warpAffine(np.dot(np.dot(calib['P1gr'], orig_im[:, :, 1]), calib['Q1gr'].T), M, (rotate_shape[1], rotate_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    sim_fc[::2, ::2] = cv2.warpAffine(np.dot(np.dot(calib['P1b'], orig_im[:, :, 2]), calib['Q1b'].T), M, (rotate_shape[1], rotate_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # sim_fc[1::2, 1::2] = np.dot(np.dot(calib['P1r'], orig_im[:, :, 0]), calib['Q1r'].T)
    # sim_fc[::2, 1::2] = np.dot(np.dot(calib['P1gb'], orig_im[:, :, 1]), calib['Q1gb'].T)
    # sim_fc[1::2, ::2] = np.dot(np.dot(calib['P1gr'], orig_im[:, :, 1]), calib['Q1gr'].T)
    # sim_fc[::2, ::2] = np.dot(np.dot(calib['P1b'], orig_im[:, :, 2]), calib['Q1b'].T)
    # print("simulate_flatcam", time.time() - start)

    # orig_shape = orig_im.shape
    # sim_fc[1::2, 1::2] = imrotate(np.dot(np.dot(calib['P1r'], orig_im[:, :, 0]), calib['Q1r'].T), -calib['angle'], mode='wrap', order=2,reshape=False)
    # sim_fc[::2, 1::2] = imrotate(np.dot(np.dot(calib['P1gb'], orig_im[:, :, 1]), calib['Q1gb'].T), -calib['angle'], mode='wrap', order=2,reshape=False)
    # sim_fc[1::2, ::2] = imrotate(np.dot(np.dot(calib['P1gr'], orig_im[:, :, 1]), calib['Q1gr'].T), -calib['angle'], mode='wrap', order=2,reshape=False)
    # sim_fc[::2, ::2] = imrotate(np.dot(np.dot(calib['P1b'], orig_im[:, :, 2]), calib['Q1b'].T), -calib['angle'], mode='wrap',  order=2,reshape=False)
    return sim_fc
def add_noise(flat_img, noise_level = 20):
    # add noise on the flatcam measurement
    # noise_level = 20db
    # calculate the average intensity of the flatcam measurement
    # flat_img_min = np.min(flat_img)
    # flat_img -= flat_img_min
    flat_var = np.var(flat_img)
    # avg_intensity = np.mean(flat_img)
    noise = np.random.normal(0, 1, flat_img.shape) * np.sqrt((10 ** (-noise_level / 10)) * flat_var)
    flat_img = flat_img + noise 
    # print(avg_intensity)
    return flat_img



def downsample_calib(calib):
    clean_calib(calib)
    for key in calib.keys():
        print(key)
        try:
            print(calib[key].shape)
        except:
            pass
    calib['P1r'] = calib['P1r'][::2, ::2]
    calib['P1gb'] = calib['P1gb'][::2, ::2]
    calib['P1gr'] = calib['P1gr'][::2, ::2]
    calib['P1b'] = calib['P1b'][::2, ::2]
    calib['Q1r'] = calib['Q1r'][::2, ::2]
    calib['Q1gb'] = calib['Q1gb'][::2, ::2]
    calib['Q1gr'] = calib['Q1gr'][::2, ::2]
    calib['Q1b'] = calib['Q1b'][::2, ::2]
    calib['cSize'][0] = int(calib['cSize'][0] / 2)
    calib['cSize'][1] = int(calib['cSize'][1] / 2)
    obtain_calib_svd(calib)
    # calib['UL_all'] = calib['UL_all'][::2, ::2]
    # calib['DL_all'] = calib['DL_all'][::2]
    # # calib['VL_all'] = calib['VL_all'][::2]
    # # calib['singL_all'] = calib['singL_all'][::2]
    # calib['UR_all'] = calib['UR_all'][::2, ::2]
    # calib['DR_all'] = calib['DR_all'][::2]
    # calib['VR_all'] = calib['VR_all'][::2]
    # calib['singR_all'] = calib['singR_all'][::2]


    return calib