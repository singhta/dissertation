
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import struct
import torch
import glob
transforms_ = [ transforms.Resize((256,256), Image.BICUBIC),
                #transforms.RandomCrop(256),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
transforms_B = [ transforms.Resize((256,256), Image.BICUBIC),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
transform = transforms.Compose(transforms_)
transformB = transforms.Compose(transforms_B)
def load_data(nr_of_channels, videosequence, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False,
              REAL_LABEL=1.0):

    trainB_path = os.path.join('datasets', subfolder, 'trainA')
    testB_path = os.path.join('datasets', subfolder, 'testA')

    video_frames_path = os.path.join('Videos', 'Frames', str(videosequence))

    frame_names = os.listdir(video_frames_path)

    trainB_image_names = os.listdir(trainB_path)

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(video_frames_path, trainB_path, frame_names, trainB_image_names,
                             batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)


def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0, y, x] = struct.unpack('f', f.read(4))[0]
                flow[1, y, x] = struct.unpack('f', f.read(4))[0]
    return flow


class data_sequence(Dataset):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B,
                 batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.frames = []
        self.train_B = []
        self.frames_path = trainA_path
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.frames.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(len(self.frames)) - 1

    def __getitem__(self,
                    idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:

        frameT_path = os.path.join(self.frames_path, "frame" + str(idx + 1) + ".jpg")
        frameT_1_path = os.path.join(self.frames_path, "frame" + str(idx) + ".jpg")
        flow_path = os.path.join(self.frames_path, "flow", str(idx) + "_" + str(idx + 1) + "_forward.flo")
        mask_path = os.path.join(self.frames_path, "flow", "reliable_" + str(idx) + "_" + str(idx + 1) + ".pgm")
        frame_T = Image.open(frameT_path)
        frame_T_1 = Image.open(frameT_1_path)
        flow = read_flow_file(flow_path)
        mask = cv2.imread(mask_path)
        zero = np.where((mask[:, :, 0] < 254) & (mask[:, :, 1] < 254) & (mask[:, :, 2] < 254))
        one = np.where((mask[:, :, 0] >= 254) & (mask[:, :, 1] >= 254) & (mask[:, :, 2] >= 254))
        mask[one] = (1, 1, 1)
        mask[zero] = (0, 0, 0)
        index_B = np.random.randint(len(self.train_B))
        real_B = Image.open(self.train_B[index_B])
        return {'frame_T':transform(frame_T), 'frame_T_1':transform(frame_T_1), 'flow':flow,
                'mask':mask, 'real_B':transformB(real_B)}

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        print(os.path.join(root, '%sA' % mode))
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))