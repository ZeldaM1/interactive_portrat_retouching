import torch
import functools
import torch.nn as nn
import models.archs.inter_base_arch as inter_base_arch
import models.archs.inter_inter_arch_v2 as inter_inter_arch_v2




class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

 
    if which_model == 'Auto_base_net':
        netG = inter_base_arch.Auto_base_net(ch_in=opt_net['in_nc'], base_nf=opt_net['base_nf'], ch_out=opt_net['out_nc'], use_psp = opt_net['use_psp'],
        layer_size=opt_net['layer_size'], mask_constrain=opt_net['mask_constrain'], use_HIN=opt_net['use_HIN'], norm_type=opt_net['norm_type'], up_sample=opt_net['up_sample'],
        final_weight = opt_net['final_weight'],spp_loc=opt_net['spp_loc'], spp_type=opt_net['spp_type'], 
        vgg_version=opt_net['vgg_version'], return_x16=opt_net['return_x16'], with_skip=opt_net['with_skip'], atten_skip=opt_net['atten_skip'])

    elif which_model == 'Auto_inter_net':
        netG = inter_inter_arch_v2.Auto_inter_net_v2_cat_corre(ch_in=opt_net['in_nc'], base_nf=opt_net['base_nf'], ch_out=opt_net['out_nc'], use_psp = opt_net['use_psp'],
        layer_size=opt_net['layer_size'], mask_constrain=opt_net['mask_constrain'], use_HIN=opt_net['use_HIN'], norm_type=opt_net['norm_type'], up_sample=opt_net['up_sample'],
        final_weight = opt_net['final_weight'],spp_loc=opt_net['spp_loc'], spp_type=opt_net['spp_type'], 
        vgg_version=opt_net['vgg_version'], return_x16=opt_net['return_x16'], with_skip=opt_net['with_skip'], atten_skip=opt_net['atten_skip'],
        norm_radius=opt_net['norm_radius'], use_disks=opt_net['use_disks'], click_only=opt_net['click_only'],  #  click options
        match_kernel=opt_net['match_kernel'],temperature=opt_net['temperature'])    
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
