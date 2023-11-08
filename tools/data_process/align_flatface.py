# do the face alignment for the flat face dataset

data_root = '/mnt/workspace/RawSense/data/flatface/fc_recon'
output_root = '/mnt/workspace/RawSense/data/flatface/fc_recon_aligned'
import face_alignment
from skimage import io
import os
from skimage import transform as trans
import cv2
import numpy as np
def align_face(rimg, landmark):
    image_size = (112,112)
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    src[:,0] += 8.0
    assert landmark.shape[0]==68 or landmark.shape[0]==5
    assert landmark.shape[1]==2
    if landmark.shape[0]==68:
      landmark5 = np.zeros( (5,2), dtype=np.float32 )
      landmark5[0] = (landmark[36]+landmark[39])/2
      landmark5[1] = (landmark[42]+landmark[45])/2
      landmark5[2] = landmark[30]
      landmark5[3] = landmark[48]
      landmark5[4] = landmark[54]
    else:
      landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2,:]
    img = cv2.warpAffine(rimg,M,(image_size[1],image_size[0]), borderValue = 0.0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_flip = np.fliplr(img)
    return img

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
person_list = os.listdir(data_root)
person_list.sort()
for person in person_list:
    if os.path.isdir(os.path.join(data_root, person)):
        img_list = os.listdir(os.path.join(data_root, person))
        img_list.sort()
        for img in img_list:
            output_path = os.path.join(output_root, person, img)
            if os.path.exists(output_path):
                continue
            img_path = os.path.join(data_root, person, img)
            print(img_path)
            input = io.imread(img_path)
            preds = fa.get_landmarks(input)
            if preds is not None:
                face = align_face(input, preds[0])
               
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                io.imsave(output_path, face)
            else:
                print('no face detected in {}'.format(img_path))
                io.imsave(output_path, input)
