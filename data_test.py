from __future__ import print_function

import os, scipy.misc
from glob import glob
import numpy as np 
import h5py
from os import listdir
from PIL import Image
import random
import torch.utils.data as data

import torch

from summary import VisdomSummary

class CelebA(data.Dataset):
    def __init__(self, celeba_dir, img_dir, vissim):
        self.resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64',
                           'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
        self.celeba_dir = celeba_dir
        self.img_dir = img_dir
        self.vissum=vissum
        # Create dictionnary data about identities
        identities = {}
        with open(os.path.join(celeba_dir, 'Anno', 'identity_CHQ.txt'), 'rt') as file:
            for line in file:
                hq_idx, identity, iname = line.split()
                if identity not in identities:
                    identities[identity] = [iname]
                else:
                    identities[identity].append(iname)
        self.upper = dict([(k, v)
                           for k, v in identities.items() if len(v) > 7])
        self.identity = sorted(list(self.upper.keys()))

    def __call__(self, batch_size, size, level=None):
        key = 'data{}x{}'.format(size, size)
        idx = random.sample(self.identity, batch_size)
        flist1, flist2 = [], []
        batch_x1 = np.ndarray(
            (batch_size, 3, size, size), dtype=np.float32)
        batch_x2 = np.ndarray(
            (batch_size, 3, size, size), dtype=np.float32)
        x = 0
        for i in idx:
            print(x, batch_size)
            img_list = self.upper[i]
            img2_list = img_list.copy()
            img1_fn = random.choice(img_list)
            img2_list.remove(img1_fn)
            img2_fn = random.choice(img2_list)

            img1 = np.array(Image.open(os.path.join(
                self.img_dir, img1_fn)).resize((32,32))).transpose(2, 0, 1)
            print(img1.shape)
            
            # print(img1/127.5-1.0)
            # print(img1/255)
            # exit()
            
            # img2 = np.array(Image.open(os.path.join(
            #     self.img_dir, img2_fn)).resize((4, 4))).transpose(2, 0, 1)
            # rimg1 = np.resize(img1, (3, size, size))/127.5-1.0
            # rimg2 = np.resize(img1, (3, size, size))/127.5-1.0

            # rimg1 = 
            img1 = torch.tensor(img1)
            self.vissum.image2d('original', 'ori', img1)

            rimg=torch.tensor(img1/127.5-1.0)
            self.vissum.image2d('resize', 'resize', rimg)
            x += 1
        print("finish making data")

        level=3.45
        if level is not None:
            if level != int(level):
                min_lw, max_lw = int(level+1)-level, level-int(level)
                rbatch_x1 = np.ndarray(
                    (batch_size, 3, size//2, size//2), dtype=np.float32)
                rbatch_x2 = np.ndarray(
                    (batch_size, 3, size//2, size//2), dtype=np.float32)
                for i in range(x):
                    img1 = np.array(Image.open(os.path.join(
                        self.img_dir, flist1[i]))).transpose(2, 0, 1)
                    img2 = np.array(Image.open(os.path.join(
                        self.img_dir, flist2[i]))).transpose(2, 0, 1)
                    rimg1 = np.resize(img1, (3, size//2, size//2))/127.5-1.0
                    rimg2 = np.resize(img1, (3, size//2, size//2))/127.5-1.0
                    rbatch_x1[i], rbatch_x2[i] = rimg1, rimg2

                low_resol_batch_x1 = rbatch_x1.repeat(2, axis=2).repeat(2, axis=3)
                low_resol_batch_x2 = rbatch_x2.repeat(2, axis=2).repeat(2, axis=3)
                batch_x1 = batch_x1 * max_lw + low_resol_batch_x1 * min_lw
                batch_x2 = batch_x2 * max_lw + low_resol_batch_x2 * min_lw
        print("Fine")
        return batch_x1, batch_x2

    def __len__(self):
        return len(self.identity)


if __name__ == "__main__":
    celeba_dir = './datasets/celeba'
    img_dir = './datasets/CelebA-HQ'
    vissum = VisdomSummary(port=10002, env='data-test')

    celeba = CelebA(celeba_dir, img_dir, vissum)
    celeba(32, 4, 4)
    # print(celeba[0].data[0])
