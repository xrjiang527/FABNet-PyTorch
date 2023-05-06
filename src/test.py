import torch
from data import create_dataloader
from data import create_dataset_val
import  os
import imageio

from option import args
from train.train_model import Model
from utils.utils import *


def main():
    # random seed
    torch.manual_seed(args.seed)


    val_set = create_dataset_val(args, train = False)
    val_loader = create_dataloader(val_set, args, train = False)

    model = Model(args)
    psnr_record = []
    psnr_list = []
    ssim_list = []
    sr_list = []
    path_list = []
          
    for iter, batch in enumerate(val_loader):
            model.set_input(batch)
            model.test(model.network)
            visuals = model.get_current_visual(args)
            psnr , ssim = calc_metrics(visuals['SR'], visuals['HR'], crop_border=args.scale)
            sr_list.append(visuals['SR'])
            path_list.append(os.path.basename(batch['HR_path'][0].replace('HR', 'vdsr')))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    logging.info('epoch:{:5d} psnr:{:.6f} ssim:{:.5f}) '.format(1, sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))
    psnr_record.append(sum(psnr_list)/len(psnr_list))

    save_img_path = os.path.join(args.save_dir, 'results/SR/', args.data_test)
    if not os.path.exists(save_img_path): os.makedirs(save_img_path)
    for img, name in zip(sr_list, path_list):
        imageio.imwrite(os.path.join(save_img_path, name), img)
       
if __name__ == '__main__':
    main()


