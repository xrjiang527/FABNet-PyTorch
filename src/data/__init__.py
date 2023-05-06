import torch.utils.data
from data.LRHR_dataset_train import LRHRDataset as D_train
from data.LRHR_dataset_val import LRHRDataset as D_val


def create_dataloader(dataset, args, train ):
    if train:
        batch_size = args.batch_size
        shuffle = True
        num_workers = args.n_threads
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def create_dataset_train(args, train):
    dataset = D_train(args, train)
    print('===> [%s] Dataset is created.')
    return dataset

def create_dataset_val(args, train):
    dataset = D_val(args, train)
    print('===> [%s] Dataset is created.')
    return dataset
