import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''
    def __init__(self, args, train):
        super(LRHRDataset, self).__init__()
        self.train = train
        self.scale = args.scale
        self.paths_HR, self.paths_LR = None, None
        self.data_type = args.data_type
        self.dataroot_HR = args.dataroot_HR_val
        self.dataroot_LR = args.dataroot_LR_val
        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 1
        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.data_type, self.dataroot_HR)
        self.paths_LR = common.get_image_paths(self.data_type, self.dataroot_LR)
        self.noise = args.noise
        self.rgb_range = args.rgb_range

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)    
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.rgb_range)
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, self.data_type)
        hr = common.read_img(hr_path, self.data_type)

        return lr, hr, lr_path, hr_path

