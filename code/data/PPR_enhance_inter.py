import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import cv2
import random
from .points_sampler import MultiPointSampler
from .sample import DSample
class PPR_enhance_inter_dataset(data.Dataset):
    def __init__(self, opt):
        super(PPR_enhance_inter_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env, self.MASK_env = None, None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_MASK, self.sizes_MASK = util.get_image_paths(self.data_type, opt['dataroot_MASK'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(self.paths_GT), 'GT and LQ datasets have different number of images - {}, {}.'.format(len(self.paths_LQ), len(self.paths_GT))
            # assert len(self.paths_LQ) == len(self.paths_MASK), 'MASK and LQ datasets have different number of images - {}, {}.'.format(len(self.paths_LQ), len(self.paths_MASK))
        self.points_sampler = MultiPointSampler(opt['max_num_points'])

    def __getitem__(self, index):
        GT_path, LQ_path, MASK_path = None, None, None
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        LQ_path = self.paths_LQ[index]
        MASK_path = self.paths_MASK[index]
        img_GT = util.read_img(self.GT_env, GT_path)
        img_LQ = util.read_img(self.LQ_env, LQ_path)
        img_MASK = util.read_img(self.MASK_env, MASK_path)

        
        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, C = img_LQ.shape
           
            # randomly crop
            h_size = int(np.random.uniform(0.6, 1.0) * H)
            w_size = int(np.random.uniform(0.6, 1.0) * W)

            rnd_h = random.randint(0, max(0, H - h_size))
            rnd_w = random.randint(0, max(0, W - w_size))

            img_LQ = img_LQ[rnd_h:rnd_h + h_size, rnd_w:rnd_w + w_size, :]
            img_GT = img_GT[rnd_h:rnd_h + h_size, rnd_w:rnd_w + w_size, :]
            img_MASK = img_MASK[rnd_h:rnd_h + h_size, rnd_w:rnd_w + w_size, :]
            

            #resize to target size
            img_LQ = cv2.resize(img_LQ, (GT_size, GT_size), cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), cv2.INTER_LINEAR)
            img_MASK = cv2.resize(img_MASK, (GT_size, GT_size), cv2.INTER_LINEAR)
            


            # augmentation - flip, rotate
            img_LQ, img_GT,img_MASK = util.augment([img_LQ, img_GT, img_MASK], self.opt['use_flip'], self.opt['use_rot'])

        elif self.opt['phase'] == 'val':
            img_MASK = img_MASK[:,:,0]


        self.points_sampler.sample_object(img_MASK)
        points = np.array(self.points_sampler.sample_points())


        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            # print(img_GT.shape, img_MASK.shape)
            # img_MASK = img_MASK[:, :, [2, 1, 0]]
        if len(img_MASK.shape) ==2:
            img_MASK = img_MASK[:,:,np.newaxis]

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_MASK = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MASK, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'GT': img_GT, 'MASK':img_MASK, 'LQ_path': LQ_path, 'GT_path': GT_path, 'MASK_path': MASK_path, 'points': points.astype(np.float32)}

    def __len__(self):
        return len(self.paths_GT)
