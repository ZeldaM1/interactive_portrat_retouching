import functools
import math
import time
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.mod_resnet as mod_resnet
from models.archs.share_Unet_region_res_arch import Deshadow_DUAL_norm_with_res_skip
from .correspondence import Correlation_fuse_cat_corre
from .ops import DistMaps
 
 
class Auto_inter_net_v2_cat_corre(Deshadow_DUAL_norm_with_res_skip): 
    def __init__(self, norm_radius=5, use_disks=True, return_x16=True, click_only=False, match_kernel=3,temperature=0.01,**kwargs):
        super().__init__(return_x16=return_x16, **kwargs)

        self.correlation = Correlation_fuse_cat_corre(return_x16, click_only, match_kernel, temperature)
        self.dist_maps = DistMaps(norm_radius=norm_radius,use_disks=use_disks)
        self.click_only = click_only
        

    def forward(self, input, points=None):
        coord_features=None
        if points is not None:
            coord_features = self.dist_maps(input, points) if self.click_only else torch.cat([input, self.dist_maps(input, points)], dim=1) 

        torch.cuda.synchronize()
        t_begin = time.time()


        prev_fea = self.conv_head(input)
        blockout = prev_fea
        prev_fea_en = prev_fea
        prev_fea_list=[prev_fea]
        skip_list=[prev_fea]
        atten_list=[prev_fea]
 
        for i in range(self.layer_size//2):
            en_block = getattr(self, 'en_block_{:d}'.format(i+1)) 
            prev_fea_en = self.max_pool(prev_fea_en)
            prev_fea_list.append(prev_fea_en)
            atten, blockout = en_block(blockout, prev_fea_en)
            skip_list.append(blockout)
            atten_list.append(atten)

        if (self.use_psp and self.spp_loc == 'center'):
            if self.return_x16:
                f16, f8, f4, f2 = self.seg_enc(input)
                blockout = self.spp1(f16, f8, f4, f2, blockout)
                # print('-----------------', blockout.shape, atten.shape)
                # import pdb
                # pdb.set_trace()
            else:
                f8, f4, f2 = self.seg_enc(input)
                blockout = self.spp1(f8, f4, f2, blockout)

            if coord_features is not None:
                blockout = self.correlation(blockout, atten, coord_features)
                # print('-----------------', blockout.shape, atten.shape)

        if self.with_skip:
            for i in range(self.layer_size//2):
                de_block = getattr(self, 'de_block_{:d}'.format(i+1)) 
                if self.atten_skip:
                    atten, blockout = de_block(blockout, prev_fea_list[self.layer_size//2-1-i], atten_list[self.layer_size//2-1-i])  
                else:
                    atten, blockout = de_block(blockout, prev_fea_list[self.layer_size//2-1-i], skip_list[self.layer_size//2-1-i])  
                
        else:
            for i in range(self.layer_size//2):
                de_block = getattr(self, 'de_block_{:d}'.format(i+1)) 
                atten, blockout = de_block(blockout, prev_fea_list[self.layer_size//2-1-i])
                

        mask_out = self.conv_tail2(atten)
        spp_input =self.conv_spp(blockout)
        if (self.use_psp and self.spp_loc == 'final'):
            if self.spp_type=='single':
                spp_out = self.spp1(spp_input)
            elif self.spp_type=='multi':
                if self.return_x16:
                    f16, f8, f4, f2 = self.seg_enc(input)
                    blockout = self.spp1(f16, f8, f4, f2, blockout)
                else:
                    f8, f4, f2 = self.seg_enc(input)
                    blockout = self.spp1(f8, f4, f2, blockout)
        else:
            spp_out = spp_input
            
        if self.final_weight:
            res_out = spp_out * mask_out
        else:
            res_out = spp_out
        res_out = self.conv_tail1(res_out)
        b,c,h,w = input.shape
        res_out = F.upsample(res_out, size=(h,w), mode='bilinear')
        img_out = res_out + input
        torch.cuda.synchronize()
        t_end = time.time()

        if self.mask_constrain:
            mask_out = F.upsample(mask_out, size=(h,w), mode='bilinear')
            return img_out, mask_out, (t_end - t_begin)
        else: 
            return img_out , (t_end - t_begin)

   
  