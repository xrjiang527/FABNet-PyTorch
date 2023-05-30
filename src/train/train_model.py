
import logging
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import model.FABNet_tiny as model

class Model():

    def DataParallelModelProcess(self, model, is_eval='train', device='cuda'):
        if is_eval == 'eval':
            model.eval()
        elif is_eval == 'train':
            model.train()
        else:
            raise ValueError('is_eval should be eval or train')
        model.to(device)
        return model

    def DataParallelCriterionProcess(self, criterion, device='cuda'):
        criterion.to(device)
        return criterion

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        device = args.device
        
        # Model
        network = model.FABNet(args)
        load_model(network, args.pre_model)
        print_model_parms(network, 'student_model')
        self.network = self.DataParallelModelProcess(network, 'train', device)

        # Data
        self.Tensor = torch.cuda.FloatTensor
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None
    
        # optimizer
        self.G_solver = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.network.parameters()), 'initial_lr': args.lr}], args.lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

        self.loss = self.DataParallelCriterionProcess(nn.L1Loss())

        cudnn.benchmark = True
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    def set_input(self, batch, need_HR=True):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)
        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)

    def adjust_learning_rate(self, base_lr, optimizer, declr_rate, epoch, epoch_step):
        lr = base_lr * (declr_rate ** (epoch // epoch_step))
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def forward(self, epoch):

        self.SR = self.network.train()(self.LR, epoch)

    def backward(self):
        G_loss = 0.0     
        G_loss = G_loss + self.loss(self.SR, self.HR)       
        G_loss.backward()
        self.G_loss = G_loss.item()
    
    def optimize_parameters(self, epoch):
        self.forward(epoch)
        self.G_solver.zero_grad()
        self.backward()
        self.G_solver.step()

    def get_current_visual(self, args, need_np=True, need_HR=True):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_np:  out_dict['LR'], out_dict['SR'] = Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                args.rgb_range)
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
            if need_np: out_dict['HR'] = Tensor2np([out_dict['HR']],
                                                   args.rgb_range)[0]
        return out_dict

    def test(self, model):
        model.eval()
        with torch.no_grad():
            
            SR= model(self.LR, 1)

            if isinstance(SR, list):
                self.SR = SR[-1]
            else:
                self.SR = SR
        
        model.train()

    def print_info(self, step):
        logging.info('step:{:5d} G_lr:{:.6f} G_loss:{:.5f})'.format(step, self.G_solver.param_groups[-1]['lr'], self.G_loss))


    def save_ckpt(self):
        torch.save(self.network.state_dict(),
                   osp.join(self.args.save_dir, 'best'  + '.pth'))



