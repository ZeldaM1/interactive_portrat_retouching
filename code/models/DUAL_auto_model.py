import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
import os
logger = logging.getLogger('base')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DUAL_auto_model(BaseModel):
    def __init__(self, opt):
        super(DUAL_auto_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        
        if train_opt is not None:
            self.load_region_weight()

        self.load()

        if self.is_train:
            self.netG.train()
            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            #data
            self.use_mask = train_opt['use_mask']
            if self.use_mask:
                if train_opt['mask_criterion'] == 'bce':
                    self.cri_mask = nn.BCEWithLogitsLoss().to(self.device)
                    self.mask_w = train_opt['mask_weight']
                else:
                    raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            if train_opt['finetune_adafm']:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    v.requires_grad = False
                    if k.find('adafm') >= 0:
                        v.requires_grad = True
                        optim_params.append(v)
                        logger.info('Params [{:s}] will optimize.'.format(k))
            else:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
        self.input = self.var_L
        self.mask = data['MASK'].to(self.device)

        if self.use_mask:
            # self.mask = data['MASK'].to(self.device)
            # mask = torch.sum(mask, 1).unsqueeze(1)
            self.weights = torch.ones_like(self.mask)
            self.weights[self.mask > 0] = 5


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        self.fake_H, self.fake_mask = self.netG(self.input)
        loss_all = 0
        if self.use_mask:
            # print('----------', self.weights.shape, self.fake_H.shape, self.input.shape)
            # print('----------', self.l_pix_w, self.mask_w)

            l_pix = self.l_pix_w * self.cri_pix(self.fake_H * self.weights, self.real_H * self.weights)
            l_mask = self.mask_w * self.cri_mask(self.fake_mask, self.mask)
            loss_all+=l_pix
            loss_all+=l_mask
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            loss_all+=l_pix

        loss_all.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        if self.use_mask:
            self.log_dict['l_mask'] = l_mask.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.fake_mask = self.netG(self.input)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['fake_mask'] = torch.sigmoid(self.fake_mask).detach()[0].float().cpu()#>0.5
        out_dict['mask'] = self.mask.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_region_weight(self):
        load_path_G = self.opt['train']['pretrain']
        if load_path_G is not None:
            logger.info('Loading region aware weight for G [{:s}] ...'.format(load_path_G))
            self.load_pretrained_weights(load_path_G, self.netG)

    def update(self, new_model_dict):
        if isinstance(self.netG, nn.DataParallel):
            network = self.netG.module
            network.load_state_dict(new_model_dict)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
