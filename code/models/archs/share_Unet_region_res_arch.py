import functools
import math

from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.mod_resnet as mod_resnet
from .ops import DistMaps
from models.archs.cbam import CBAM

 

 

class SPP_v2(nn.Module):
    def __init__(self, ch_in, reduction=16, num_levels=4):
        super(SPP_v2, self).__init__()
        self.num_levels = num_levels
        self.avepool = nn.AvgPool2d(2)
        pyramid_models=[]
        for i in range(self.num_levels):
            pyramid_models.append(nn.Conv2d(ch_in, ch_in, 3, 1, 1))
        self.pyramid_models = nn.Sequential(*pyramid_models)
        self.conv = nn.Conv2d(5*ch_in, ch_in, 3, 1, 1)

    def forward(self, x):
        pyramid = [x]
        b, c, h, w = x.shape
        for i in range(self.num_levels):
            x = self.avepool(x)
            pyramid_out = self.pyramid_models[i](x)
            pyramid_out = F.interpolate(pyramid_out, size=(h, w), mode='bilinear')

            pyramid.append(pyramid_out)

        pyramid_out = torch.cat(pyramid,dim=1)
        out = self.conv(pyramid_out)
        return out

 
class SPP_center_f16(nn.Module):
    def __init__(self, input_ch=32, output_ch=32):
        super(SPP_center_f16, self).__init__()
        self.pyramid_models=nn.Sequential(nn.Conv2d(input_ch, input_ch, 3, 1, 1),
                                        nn.Conv2d(64, 32, 3, 1, 1),
                                        nn.Conv2d(64, 32, 3, 1, 1),
                                        nn.Conv2d(128, 64, 3, 1, 1),
                                        nn.Conv2d(256, 128, 3, 1, 1))
                                            
         
        self.conv = nn.Conv2d(input_ch+256, output_ch, 3, 1, 1)

    def forward(self, f16, f8, f4, f2, input):
        fea_dict={'f1':input, 'f2':f2, 'f3':f4, 'f4':f8, 'f5':f16}
        # f16 1/16, 256
        # f8 1/8, 128
        # f4 1/4, 64
        # f2 1/2 64
        # f1 1 64

        pyramid = []
        b, c, h, w = input.shape
        for i in range(5):
            pyramid_out = self.pyramid_models[i](fea_dict['f{:d}'.format(i+1)])
            pyramid_out = F.interpolate(pyramid_out, size=(h, w), mode='bilinear')
            pyramid.append(pyramid_out)

        pyramid_out = torch.cat(pyramid,dim=1)
        out = self.conv(pyramid_out)
        return out


class SPP_center(nn.Module):
    def __init__(self, input_ch=32, output_ch=32):
        super(SPP_center, self).__init__()
        self.pyramid_models=nn.Sequential(nn.Conv2d(input_ch, input_ch, 3, 1, 1),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.Conv2d(128, 128, 3, 1, 1))
                                               
         
        self.conv = nn.Conv2d(input_ch+256, output_ch, 3, 1, 1)

    def forward(self, f8, f4, f2, input):
        fea_dict={'f1':input, 'f2':f2, 'f3':f4, 'f4':f8}
        pyramid = []
        b, c, h, w = input.shape
        for i in range(4):
            pyramid_out = self.pyramid_models[i](fea_dict['f{:d}'.format(i+1)])
            pyramid_out = F.interpolate(pyramid_out, size=(h, w), mode='bilinear')
            pyramid.append(pyramid_out)

        pyramid_out = torch.cat(pyramid,dim=1)
        out = self.conv(pyramid_out)
        return out

 



class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        b,c,h,w=x.shape
        f16 = F.upsample(f16, size=(h,w), mode='nearest')
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return r, x

class Seg_Encoder(nn.Module):
    def __init__(self, return_x16=True, extr_ch=0):
        super(Seg_Encoder, self).__init__()
        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=extr_ch)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.return_x16 = return_x16
        if return_x16:
            self.layer3 = resnet.layer3
        
         
 
    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        f2 = self.relu(x)   # 1/2, 64
        x = self.maxpool(f2)  # 1/4, 64, 64
        f4 = self.res2(x)   # 1/4, 64, 64
        f8 = self.layer2(f4) # 1/8, 128, 32
        if self.return_x16:
            f16 = self.layer3(f8)   
            return f16, f8, f4, f2
        else: 
            return f8, f4, f2

class Enblock_DUAL(nn.Module):
    def __init__(self, ch_in, ch_base, ch_add, version=None, use_HIN=False, norm_type='instance'):
        super(Enblock_DUAL, self).__init__()
        if norm_type=='instance':
            self.use_HIN = use_HIN
            self.norm = nn.InstanceNorm2d(ch_base // 2, affine=True) if use_HIN else nn.InstanceNorm2d(ch_base, affine=True)
        elif norm_type=='batch':
            self.use_HIN = False
            self.norm = nn.BatchNorm2d(ch_base, affine=True)
        else:
            raise NotImplementedError('Normalization not recognized')


        self.conv1 = nn.Conv2d(ch_in, ch_base, 3, 2, 1)
        self.conv2 = nn.Conv2d(ch_base, ch_base, 3, 1, 1)
        self.relu_2 = nn.ReLU()
        self.fuse = FeatureFusionBlock(ch_base+ch_add, ch_base)
        self.version = version

    def forward(self, x, prev):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(conv2_out, 2, dim=1)
            conv2_out = self.relu_2(torch.cat([self.norm(out_1), out_2], dim=1))
        else:
            conv2_out =  self.relu_2(self.norm(conv2_out))
        atten, fuse_output = self.fuse(conv2_out, prev)
        if self.version == 'v2':
            fuse_output = torch.cat([fuse_output, conv2_out], dim=1)
        return atten, fuse_output


class Enblock_DUAL_vgg(nn.Module):
    def __init__(self, ch_in, ch_base, ch_add, version=None, use_HIN=False, norm_type='instance'):
        super(Enblock_DUAL_vgg, self).__init__()
        self.norm_type = norm_type
        if norm_type=='instance':
            self.use_HIN = use_HIN
            self.norm = nn.InstanceNorm2d(ch_base // 2, affine=True) if use_HIN else nn.InstanceNorm2d(ch_base, affine=True)
        elif norm_type=='batch':
            self.use_HIN = False
            self.norm = nn.BatchNorm2d(ch_base, affine=True)
        elif norm_type==None:
            pass
        else:
            raise NotImplementedError('Normalization not recognized')


        self.conv1 = nn.Conv2d(ch_in, ch_base, 3, 2, 1)
        self.conv2 = nn.Conv2d(ch_base, ch_base, 3, 1, 1)
        self.relu_2 = nn.ReLU()
        self.fuse = FeatureFusionBlock(ch_base+ch_add, ch_base)
        self.version = version

    def forward(self, x, prev):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        if not (self.norm_type==None):
            if self.use_HIN:
                out_1, out_2 = torch.chunk(conv2_out, 2, dim=1)
                conv2_out = self.relu_2(torch.cat([self.norm(out_1), out_2], dim=1))
            else:
                conv2_out =  self.relu_2(self.norm(conv2_out))
        else:
            conv2_out =  self.relu_2(conv2_out)

        atten, fuse_output = self.fuse(conv2_out, prev)
        if self.version == 'v2':
            fuse_output = torch.cat([fuse_output, conv2_out], dim=1)
        return atten, fuse_output


 
class Deblock_DUAL(nn.Module):
    def __init__(self, ch_in, ch_base, ch_add, version=None,  use_HIN=False, norm_type='instance'):
        super(Deblock_DUAL, self).__init__()
        if norm_type=='instance':
            self.use_HIN = use_HIN
            self.norm = nn.InstanceNorm2d(ch_base // 2, affine=True) if use_HIN else nn.InstanceNorm2d(ch_base, affine=True)
        elif norm_type=='batch':
            self.use_HIN = False
            self.norm = nn.BatchNorm2d(ch_base, affine=True)
        else:
            raise NotImplementedError('Normalization not recognized')

        self.conv1 = nn.ConvTranspose2d(ch_in, ch_base, 3, 2, 1)
        self.conv2 = nn.Conv2d(ch_base, ch_base, 3, 1, 1)
        self.relu_2 = nn.ReLU()
        self.fuse = FeatureFusionBlock(ch_base+ch_add, ch_base)
        self.version = version
        

    def forward(self, x, prev):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(conv2_out, 2, dim=1)
            conv2_out = self.relu_2(torch.cat([self.norm(out_1), out_2], dim=1))
        else:
            conv2_out =  self.relu_2(self.norm(conv2_out))
        
        atten, fuse_output = self.fuse(conv2_out, prev)
        if self.version =='v2':
            fuse_output = torch.cat([fuse_output, conv2_out], dim=1)

        return atten, fuse_output



class Deblock_DUAL_inter(nn.Module):
    def __init__(self, ch_in, ch_base, ch_add, version=None,  use_HIN=False, norm_type='instance'):
        super(Deblock_DUAL_inter, self).__init__()
        if norm_type=='instance':
            self.use_HIN = use_HIN
            self.norm = nn.InstanceNorm2d(ch_base // 2, affine=True) if use_HIN else nn.InstanceNorm2d(ch_base, affine=True)
        elif norm_type=='batch':
            self.use_HIN = False
            self.norm = nn.BatchNorm2d(ch_base, affine=True)
        else:
            raise NotImplementedError('Normalization not recognized')

        self.conv1 = nn.Conv2d(ch_in, ch_base, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch_base, ch_base, 3, 1, 1)
        self.relu_2 = nn.ReLU()
        self.fuse = FeatureFusionBlock(ch_base+ch_add, ch_base)
        self.version = version
        

    def forward(self, x, prev):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(conv2_out, 2, dim=1)
            conv2_out = self.relu_2(torch.cat([self.norm(out_1), out_2], dim=1))
        else:
            conv2_out = self.relu_2(self.norm(conv2_out))
        
        atten, fuse_output = self.fuse(conv2_out, prev)
        if self.version =='v2':
            fuse_output = torch.cat([fuse_output, conv2_out], dim=1)

        return atten, fuse_output


 
class Deblock_DUAL_skip(nn.Module):
    def __init__(self, ch_in, ch_base, ch_add, version=None,  use_HIN=False, norm_type='instance', up_scale='trans', skip=True):
        super(Deblock_DUAL_skip, self).__init__()
        if norm_type=='instance':
            self.use_HIN = use_HIN
            self.norm = nn.InstanceNorm2d(ch_base // 2, affine=True) if use_HIN else nn.InstanceNorm2d(ch_base, affine=True)
        elif norm_type=='batch':
            self.use_HIN = False
            self.norm = nn.BatchNorm2d(ch_base, affine=True)
        else:
            raise NotImplementedError('Normalization not recognized')

        self.up_scale = up_scale
        self.conv1 = nn.ConvTranspose2d(ch_in, ch_base, 3, 2, 1) if up_scale == 'trans' else nn.Conv2d(ch_in, ch_base, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch_base*2, ch_base, 3, 1, 1) if skip else nn.Conv2d(ch_base, ch_base, 3, 1, 1)
        self.skip = skip
        self.relu_2 = nn.ReLU()
        self.fuse = FeatureFusionBlock(ch_base+ch_add, ch_base)
        
        

    def forward(self, x, prev, skip=None):
        if self.up_scale == 'inter':
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
       
        conv1_out = self.conv1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(conv1_out, 2, dim=1)
            conv1_out = self.relu_2(torch.cat([self.norm(out_1), out_2], dim=1))
        else:
            conv1_out =  self.relu_2(self.norm(conv1_out))

        if self.skip:
            # print('------',conv1_out.shape,skip.shape )
            if not (conv1_out.shape ==skip.shape) and ((skip.shape[2]-conv1_out.shape[2])<=4):
                conv1_out = F.interpolate(conv1_out, size=skip.shape[2:], mode='nearest')
            conv1_out = torch.cat([conv1_out, skip], dim=1)

        conv2_out = self.conv2(conv1_out)
        
        atten, fuse_output = self.fuse(conv2_out, prev)
        return atten, fuse_output

 

class Deshadow_DUAL_norm_with_res_skip(nn.Module):
    def __init__(self, ch_in=3, base_nf=64, ch_out=3, layer_size=4, mask_constrain=False, use_psp=True, up_sample='trans', return_x16=True,
                 use_HIN=False, norm_type='instance', final_weight=False, spp_loc='final', spp_type='single', vgg_version=False, with_skip=False, 
                 atten_skip=False, extr_ch=0):
        super(Deshadow_DUAL_norm_with_res_skip, self).__init__()
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
                self.seg_enc = Seg_Encoder(return_x16, extr_ch)
                self.spp1 = SPP_center_f16(input_ch=base_nf//2, output_ch = base_nf//2) if return_x16 else SPP_center(input_ch=base_nf//2, output_ch = base_nf//2) 
            else: #center
                self.seg_enc = Seg_Encoder(return_x16, extr_ch)
                self.spp1 = SPP_center_f16(input_ch=out_ch_en, output_ch = out_ch_en) if return_x16 else SPP_center(input_ch=out_ch_en, output_ch = out_ch_en)

            self.spp_loc = spp_loc
            self.spp_type = spp_type
            
 
        self.conv_tail1 = nn.Conv2d(base_nf//2, ch_out, 3, 1, 1)
        self.conv_tail2 = nn.Conv2d(base_nf, 1, 3, 1, 1)
        self.return_x16 = return_x16
        self.with_skip = with_skip
        self.atten_skip = atten_skip
        
         

    def forward(self, input, points):
        
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
            else:
                f8, f4, f2 = self.seg_enc(input)
                blockout = self.spp1(f8, f4, f2, blockout)

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

        if self.mask_constrain:
            mask_out = F.upsample(mask_out, size=(h,w), mode='bilinear')
            return img_out, mask_out
        else: 
            return img_out 

 