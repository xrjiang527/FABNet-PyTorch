import torch
from data import create_dataloader
from data import create_dataset_train
from data import create_dataset_val

from option import args
from train.train_model import Model
from utils.utils import *


def main():
    # random seed
    torch.manual_seed(args.seed)

    # create train and val dataloader
    train_set_train = create_dataset_train(args, train = True)
    train_loader = create_dataloader(train_set_train, args, train = True)

    val_set = create_dataset_val(args, train = False)
    val_loader = create_dataloader(val_set, args, train = False)

    model = Model(args)
    psnr_record = []
    best = 0.0
    best_epoch = 1

    for epoch in range(1, args.epochs):
        logging.info('\n===> Training Epoch : [%d/%d]' %(epoch, args.epochs))
        
        for iter, batch in enumerate(train_loader):
                model.adjust_learning_rate(args.lr, model.G_solver, args.gamma, epoch, args.decay)
                model.set_input(batch)              
                model.optimize_parameters(epoch)
                if ( iter + 1) % args.print_every == 0:
                    model.print_info(iter)
                
                
        print('=====> Validating...')

        psnr_list = [] 
        ssim_list = []
                   
        for iter_test, batch in enumerate(val_loader):
                model.set_input(batch)
                model.test(model.network)
                visuals = model.get_current_visual(args)
                psnr, ssim = calc_metrics(visuals['SR'], visuals['HR'], crop_border=args.scale)
                        
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        logging.info('epoch:{:5d} dataset:{:5d} iter:{:5d} psnr:{:.6f} ssim:{:.5f}) '.format(epoch, iter_test, iter,
                                                                                                sum(psnr_list) / len(psnr_list),
                                                                                                sum(ssim_list) / len(ssim_list)))
        psnr_record.append(sum(psnr_list) / len(psnr_list))
                
        if best < sum(psnr_list) / len(psnr_list):
                       best_epoch = epoch
                       best = sum(psnr_list) / len(psnr_list)
                       model.save_ckpt()
                       logging.info(
                                'best_epoch:{:5d} psnr:{:.6f} ssim:{:.5f}) '.format(epoch, sum(psnr_list) / len(psnr_list),
                                                                             sum(ssim_list) / len(ssim_list)))
        else:
                        logging.info('best_epoch:{:5d} psnr:{:.6f} ssim:{:.5f}) '.format(best_epoch, best, best))

                    

if __name__ == '__main__':
    main()


