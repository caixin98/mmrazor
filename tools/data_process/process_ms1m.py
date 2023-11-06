# transfer mxnet record to png images
import mxnet as mx
from torch.utils.data import Dataset
import os
import numpy as np
import numbers
import torch
import cv2
from tqdm import tqdm
class MXFaceDataset(Dataset):
    CLASSES = 93431
    def __init__(self, root_dir, save_dir):
        super(MXFaceDataset, self).__init__()
   
        self.root_dir = root_dir
        self.save_dir = save_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        try:
            idx = self.imgidx[index]
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            label = torch.tensor(label, dtype=torch.long)
            sample = mx.image.imdecode(img).asnumpy()
        except:
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
        # save sample into png
        # assign label into name
        name = str(label.item()) + '/' + str(idx)
        
        save_path = os.path.join(self.save_dir, name +'.jpg')
        # mx.image.imsave(save_path, sample)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, sample, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return dict(img=sample, gt_label = label)

    def __len__(self):
        return len(self.imgidx)
root_dir = '/mnt/workspace/RawSense/data/ms1m-retinaface-t1'
save_dir = '/mnt/workspace/RawSense/data/ms1m-jpg'
dataset = MXFaceDataset(root_dir, save_dir)
# for i in tqdm(range(len(dataset))):
#     dataset[i]
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)

for data in tqdm(loader):
    pass