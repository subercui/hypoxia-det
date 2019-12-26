import getopt
import sys
from os import listdir
import os
import random
import torch
from torch.utils import data
import glob
import numpy as np
import scipy.misc
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DRIVEData(data.Dataset):
    '''
    Finish the implementation of our custom dataloader class.
    We will use the test images as our validation set.
    '''

    def __init__(self,
                 root,
                 split='train',
                 patch_size=64,
                 scale_factor=255
                 ):
        self.ROOT_PATH = root
        self.split = split
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.TRAIN_IMG_PATH = os.path.join(root, 'training', 'img')
        self.TRAIN_SEG_PATH = os.path.join(root, 'training', 'mask')
        self.TEST_IMG_PATH = os.path.join(root, 'testing', 'img')
        self.TEST_SEG_PATH = os.path.join(root, 'testing', 'mask')
        if split == 'train':
            img_path, seg_path = self.TRAIN_IMG_PATH, self.TRAIN_SEG_PATH
        elif split == 'test':
            img_path, seg_path = self.TEST_IMG_PATH, self.TEST_SEG_PATH
        else:
            raise ValueError
        self.img_files = listdir(img_path)
        self.mask_files = listdir(seg_path)

    def __len__(self):
        return len(self.img_files)

    def select_patch(self, pixels_img, pixels_seg, size, width, height):
        # select coordinates for patch
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)

        return pixels_img[x:x+size, y:y+size, :], pixels_seg[x:x+size, y:y+size]

    def concat_patch(self, pixels_img, pixels_seg, size=256):
        # size = patch_size
        # break down the image into patches of desired size and stack the patches
        # for example, image of (512,512,3) => (4, 256, 256, 3)
        factor = int(512/size)
        img_concat = np.zeros((factor**2, size, size, 3))
        seg_concat = np.zeros((factor**2, size, size))
        for i in range(0, factor):
            for j in range(0, factor):
                img_concat[i*factor+j, :, :, :] = pixels_img[size *
                                                             i:size*(i+1), size*j:size*(j+1), :]
                seg_concat[i*factor+j, :, :] = pixels_seg[size *
                                                          i:size*(i+1), size*j:size*(j+1)]
        return img_concat, seg_concat

    def __getitem__(self, i):  # i is index
        assert self.split in ['train', 'test', 'result']

        if self.split == 'train':
            full_img_path = os.path.join(
                self.TRAIN_IMG_PATH, self.img_files[i])
            full_seg_path = os.path.join(
                self.TRAIN_SEG_PATH, self.img_files[i].replace('HE-green', 'EF5'))

        else:  # otherwise split is 'test'
            full_img_path = os.path.join(self.TEST_IMG_PATH, self.img_files[i])
            full_seg_path = os.path.join(
                self.TEST_SEG_PATH, self.img_files[i].replace('HE-green', 'EF5'))

        im_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        im_seg = cv2.imread(full_seg_path, cv2.IMREAD_GRAYSCALE)
        assert im_img.ndim == 2
        assert isinstance(im_img, np.ndarray)
        assert im_img.shape == im_seg.shape
        width, height = im_img.shape
        im_img = im_img[:, :, None]

        if self.split == 'train' or self.split == 'test':
            # normalize to [0,1]
            pixels_img = np.array(im_img)/255
            pixels_seg = np.array(im_seg)/self.scale_factor
            # make sure there is something in the label img
            assert pixels_seg.max() > 60/self.scale_factor
            patch_img, patch_seg = self.select_patch(
                pixels_img, pixels_seg, self.patch_size, width, height)
            # do not patch the background
            # while patch_img.mean() < (7.6 / 255):
            #     # print('do not patch the background\n')
            #     patch_img, patch_seg = self.select_patch(
            #         pixels_img, pixels_seg, self.patch_size, width, height)

            img = torch.tensor(patch_img).permute(2, 0, 1)
            seg = torch.tensor(patch_seg).unsqueeze(0)
        else:
            pass
            # patch_img = im_img.resize((512,512)) #resample = NEAREST
            # patch_seg = im_seg.resize((512,512))
            # pixels_img = np.array(patch_img)/255
            # pixels_seg = np.array(patch_seg)/255
            # # for eval reconstruction
            # # make sure the input patch_size to concat_patch() equals args.patch_size
            # # which corresponds to the patch size model was trained on
            # patch_img, patch_seg = self.concat_patch(pixels_img, pixels_seg, 256)
            # img = torch.tensor(patch_img).permute(0,3,1,2)
            # seg = torch.tensor(patch_seg).unsqueeze(1)

        '''
        I have done the pipelining, your task is to:

        1. Load the imgs to numpy arrays.
            Hint: Use the PIL.Image library for this.
            Note: img should have shape 3xHxW
                  seg should have shape HxW. Furthermore,
                  each pixel in seg should have a value indexing
                  which class it is apart of. For this dataset,
                  we are segmenting the blood vessels from the
                  background class. Thus, there are only two classes,
                  0 = background class, 1 = blood vessel class.

        2. Normalize the images to [0, 1] or z-score norm.
            Hint: if you z-score normalize, you should consider
            whether to normalize each RGB channel seperately or
            normalize all channels simultaneously.

        3. Rescale the images to target size.
            Note: target size is your choice, but
            it should be appropriate depending on your
            model of choice. I.e, some models require
            every dimension to be divisible by 16 due
            to pooling layers. Also the choice of interpolation
            matters.
            Hint: seg only has only viable interpolation method
            (since the values are discrete, {0, 1}) while img has a few options.

        4. Apply augmentations here.

        5. Turn the numpy arrays into torch tensors.
        '''

        data_dict = {
            "name": self.img_files[i],
            "image": img,
            "mask": seg,
            # "class": diagnosis
        }

        return data_dict


if __name__ == '__main__':
    dataset = DRIVEData(
        root='/home/haotian/Code/vessel_segmentation/data/hypoxia img')
