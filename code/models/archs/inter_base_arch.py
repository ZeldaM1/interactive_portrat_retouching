import functools
import math

from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.mod_resnet as mod_resnet
from models.archs.share_Unet_region_res_arch import Seg_Encoder, Enblock_DUAL, Enblock_DUAL_vgg, Deblock_DUAL, Deblock_DUAL_inter, Deblock_DUAL_skip
 


class SPP_resonly_center_f16(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid_models=nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                        nn.Conv2d(64, 32, 3, 1, 1),
                                        nn.Conv2d(128, 64, 3, 1, 1),
                                        nn.Conv2d(256, 128, 3, 1, 1))
                                            
         
        self.conv = nn.Conv2d(256, 128, 3, 1, 1)

    def forward(self, f16, f8, f4, f2):
        fea_dict={'f1':f2, 'f2':f4, 'f3':f8, 'f4':f16}
        # f16 1/16, 256
        # f8 1/8, 128
        # f4 1/4, 64
        # f2 1/2 64
    

        pyramid = []
        b, c, h, w = f8.shape
        for i in range(4):
            pyramid_out = self.pyramid_models[i](fea_dict['f{:d}'.format(i+1)])
            pyramid_out = F.interpolate(pyramid_out, size=(h, w), mode='bilinear')
            pyramid.append(pyramid_out)

        pyramid_out = torch.cat(pyramid,dim=1)
        out = self.conv(pyramid_out)
        return out

 

 

class Deshadow_DUAL_norm_with_res_skip_base(nn.Module):
    def __init__(self, ch_in=3, base_nf=64, ch_out=3, layer_size=4, mask_constrain=False, use_psp=True, up_sample='trans', return_x16=True,
                 use_HIN=False, norm_type='instance', final_weight=False, spp_loc='final', spp_type='single', vgg_version=False, with_skip=False, 
                 atten_skip=False, extr_ch=0):
        super(Deshadow_DUAL_norm_with_res_skip_base, self).__init__()
        self.final_weight = final_weight
        self.layer_size =layer_size
        self.conv_head = nn.Conv2d(ch_in, base_nf, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(2)
        self.mask_constrain = mask_constrain
        self.use_psp = use_psp
        
        for i in range(self.layer_size//2):
            en_name = 'en_block_{:d}'.format(i+1)
            de_name = 'de_block_{:d}'.format(self.layer_size//2 - i)

            in_ch_en = base_nf if (i==0) else ((2**(i-1)) * base_nf)
            in_ch_de = ((2**i) * base_nf)
            out_ch_de = in_ch_en
            out_ch_en = in_ch_de
            
            if vgg_version:
                setattr(self, en_name, Enblock_DUAL_vgg(in_ch_en, out_ch_en, base_nf,  None, use_HIN=False, norm_type=None))
            else:
                setattr(self, en_name, Enblock_DUAL(in_ch_en, out_ch_en, base_nf,  None, use_HIN, norm_type))

            setattr(self, de_name, Deblock_DUAL_skip(in_ch_de, out_ch_de, base_nf, None, use_HIN, norm_type, up_sample, with_skip))
                
        
        self.conv_spp = nn.Conv2d(base_nf, base_nf//2, 1, 1)
 
        if self.use_psp:
            if (spp_loc=='final' and spp_type=='single'):
                self.spp1 = SPP_v2(base_nf//2, reduction=8, num_levels=4)
            elif (spp_loc=='final' and spp_type=='multi'):
                self.seg_enc = Seg_Encoder(return_x16, extr_ch)#input_ch=base_nf//2, output_ch = base_nf//2
                self.spp1 = SPP_resonly_center_f16() if return_x16 else SPP_center(input_ch=base_nf//2, output_ch = base_nf//2) 
            else: #center
                self.seg_enc = Seg_Encoder(return_x16, extr_ch)#input_ch=out_ch_en, output_ch = out_ch_en
                self.spp1 = SPP_resonly_center_f16() if return_x16 else SPP_center(input_ch=out_ch_en, output_ch = out_ch_en)

            self.spp_loc = spp_loc
            self.spp_type = spp_type
            
 
        self.conv_tail1 = nn.Conv2d(base_nf//2, ch_out, 3, 1, 1)
        self.conv_tail2 = nn.Conv2d(base_nf, 1, 3, 1, 1)
        self.return_x16 = return_x16
        self.with_skip = with_skip
        self.atten_skip = atten_skip
        
         
  

class Auto_base_net(Deshadow_DUAL_norm_with_res_skip_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pyr_fea_conv = nn.Conv2d(256+128, 256, 3, 1, 1)
 
    def forward(self, input):
        
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

        if (self.use_psp and self.spp_loc == 'center'):#use this
            if self.return_x16:
                f16, f8, f4, f2 = self.seg_enc(input)
                pyr_out = self.spp1(f16, f8, f4, f2)
                # print('-----------------',pyr_out.shape, blockout.shape)
                blockout = self.pyr_fea_conv(torch.cat([pyr_out, blockout], dim=1))
                
            else:
                f8, f4, f2 = self.seg_enc(input)
                pyr_out =  self.spp1(f8, f4, f2)
                blockout = self.pyr_fea_conv(torch.cat([pyr_out, blockout], dim=1))

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
                    pyr_out = self.spp1(f16, f8, f4, f2)
                    blockout = self.pyr_fea_conv(torch.cat([pyr_out, blockout], dim=1))
                
                else:
                    f8, f4, f2 = self.seg_enc(input)
                    pyr_out =  self.spp1(f8, f4, f2)
                    blockout = self.pyr_fea_conv(torch.cat([pyr_out, blockout], dim=1))
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

        if self.mask_constrain:
            mask_out = F.upsample(mask_out, size=(h,w), mode='bilinear')
            return img_out, mask_out
        else: 
            return img_out 

  