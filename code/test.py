import os.path as osp
import logging
import argparse
from collections import OrderedDict
import torch
import cv2
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from vis import draw_probmap, draw_points
import numpy as np
import os

def save_visualization_inter(input_img, points, gt_mask, gt_img, sr_img, fake_mask):
    max_interactive_points=points.shape[0]//2

    image_with_points = draw_points(input_img, points[:max_interactive_points], (0, 255, 0))
    image_with_points = draw_points(image_with_points, points[max_interactive_points:], (0, 0, 255))
 
    # gt_mask = np.repeat(gt_mask[:,:,np.newaxis], 3, axis=2)
    # predicted_mask = np.repeat(fake_mask[:,:,np.newaxis], 3, axis=2)
    # output_image = np.hstack((image_with_points, predicted_mask, gt_mask, sr_img))
    output_image = np.hstack((image_with_points, sr_img, gt_img))

    return output_image

def save_visualization(input_img, gt_mask, gt_img, sr_img, fake_mask):
    output_image = np.hstack((input_img, sr_img, gt_img))

    return output_image

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-models', type=str, default='../ckpt/c_ckpt.pth', help='Path to options YMAL file.')
parser.add_argument('--save_results', action='store_true', help='option to save results')

args = parser.parse_args()
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

if args.models is None:
    util.mkdirs(
        (path for key, path in opt['path'].items()
        if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        # print('------------------------',dataset_dir)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []

        for data in test_loader:
            need_GT = True
            model.use_mask=False
            model.feed_data(data, need_GT=need_GT)

            img_path = data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)
            
            sr_img = util.tensor2img(visuals['rlt'])  # uint8
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
                save_msk_img_path = osp.join(dataset_dir, img_name + '_mask.png')
            util.save_img(sr_img, save_img_path)



            if args.save_results:
                # print('save results')
                gt_img = util.tensor2img(visuals['GT'])  # uint8
                gt_mask = util.tensor2img(visuals['mask'])
                fake_mask = util.tensor2img(visuals['fake_mask'])
                input_img = util.tensor2img(visuals['LQ'])
                points = visuals['points']
                if points==None:
                    vis = save_visualization(input_img, gt_mask, gt_img, sr_img, fake_mask)
                else:
                    vis = save_visualization_inter(input_img, points, gt_mask, gt_img, sr_img, fake_mask)
                

                suffix = opt['suffix']
                if suffix:
                    save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = osp.join(dataset_dir, img_name + '.png')
                    save_msk_img_path = osp.join(dataset_dir, img_name + '_mask.png')
                util.save_img(vis, save_img_path)

    

            # # calculate PSNR and SSIM
            # if need_GT:
            #     gt_img = util.tensor2img(visuals['GT'])
            #     gt_mask = util.tensor2img(visuals['mask'])
            #     gt_mask = cv2.resize(gt_mask, (gt_img.shape[1], gt_img.shape[0]))
            #     sr_img = cv2.resize(sr_img, (gt_img.shape[1], gt_img.shape[0]))
            #     # print('---')
            #     psnr = util.calculate_psnr(sr_img, gt_img)
            #     mask_mse, mask_psnr = util.calculate_mask_psnr(sr_img, gt_img, gt_mask)
            #     ssim = util.calculate_ssim(sr_img, gt_img)
            #     test_results['psnr'].append(psnr)
            #     test_results['ssim'].append(ssim)
            #     # test_results['mask_mse'].append(mask_mse)
            #     # test_results['mask_psnr'].append(mask_psnr)
            #     logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            # else:
            #     logger.info(img_name)

        # if need_GT:  # metrics
        #     # Average PSNR/SSIM results
        #     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        #     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        #     ave_mask_mse = sum(test_results['mask_mse']) / len(test_results['mask_mse'])
        #     ave_mask_psnr = sum(test_results['mask_psnr']) / len(test_results['mask_psnr'])
        #     logger.info(
        #         '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; mask_mse: {:.6f}; mask_psnr: {:.6f}\n'.format(
        #             test_set_name, ave_psnr, ave_ssim, ave_mask_mse, ave_mask_psnr))
else:
    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)

    # all_models = str(args.models).split(',')
    sh = None
    model_name=os.path.basename(args.models).split('.')[0]
    opt['path']['pretrain_model_G'] = args.models 
    opt['name'] = model_name 
    opt['datasets']['val']['name'] = 'PPR10K_{}'.format(model_name)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.remove_logger('base', sh)
    sh = util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    


    model = create_model(opt)
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['mask_mse'] = []
        test_results['mask_psnr'] = []

        for data in test_loader:
            
            need_GT = True
            model.use_mask=False
            model.feed_data(data, need_GT=need_GT)

            img_path = data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)
            
            sr_img = util.tensor2img(visuals['rlt'])  # uint8
            fake_mask = util.tensor2img(visuals['fake_mask'])
        

            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
                save_msk_img_path = osp.join(dataset_dir, img_name + '_mask.png')
            util.save_img(sr_img, save_img_path)
            # util.save_img(fake_mask, save_msk_img_path)
    
            if args.save_results:
                # print('save results')
                gt_img = util.tensor2img(visuals['GT'])  # uint8
                gt_mask = util.tensor2img(visuals['mask'])
                fake_mask = util.tensor2img(visuals['fake_mask'])
                input_img = util.tensor2img(visuals['LQ'])
                points = visuals['points']
                if points==None:
                    vis = save_visualization(input_img, gt_mask, gt_img, sr_img, fake_mask)
                else:
                    vis = save_visualization_inter(input_img, points, gt_mask, gt_img, sr_img, fake_mask)
                

                suffix = opt['suffix']
                if suffix:
                    save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = osp.join(dataset_dir, img_name + '.png')
                    save_msk_img_path = osp.join(dataset_dir, img_name + '_mask.png')
                util.save_img(vis, save_img_path)
              
        

            if need_GT:
                gt_img = util.tensor2img(visuals['GT'])
                sr_img = cv2.resize(sr_img, (gt_img.shape[1], gt_img.shape[0]))
                gt_mask = util.tensor2img(visuals['mask'])
                gt_mask = cv2.resize(gt_mask, (gt_img.shape[1], gt_img.shape[0]))
                # print('---')
                psnr = util.calculate_psnr(sr_img, gt_img)
                mask_mse, mask_psnr = util.calculate_mask_psnr(sr_img, gt_img, gt_mask)
                ssim = util.calculate_ssim(sr_img, gt_img)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['mask_mse'].append(mask_mse)
                test_results['mask_psnr'].append(mask_psnr)
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; mask_mse: {:.6f}; mask_psnr: {:.6f}.'.format(img_name, psnr, ssim,mask_mse,mask_psnr))
            else:
                logger.info(img_name)

        if need_GT:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            ave_mask_mse = sum(test_results['mask_mse']) / len(test_results['mask_mse'])
            ave_mask_psnr = sum(test_results['mask_psnr']) / len(test_results['mask_psnr'])
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; mask_mse: {:.6f}; mask_psnr: {:.6f}\n'.format(
                    test_set_name, ave_psnr, ave_ssim, ave_mask_mse, ave_mask_psnr))

