from utils.augmentations import SSDAugmentation
from multibox_loss import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np

from tqdm import tqdm
from eval import eval_func

from torch.nn.parallel.data_parallel import DataParallel
from data import VOCDetection, BaseTransform, VOCAnnotationTransform
import os.path as osp

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

dataset_root = osp.join("/VOCdevkit")
basenet='vgg16_reducedfc.pth'
batch_size=64

start_iter=0
num_workers=4
lr=1e-3
momentum=0.9
weight_decay=5e-4
gamma=0.1
save_folder='weights'
MEANS = (104, 117, 123)
cfg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

dataset = VOCDetection(root=dataset_root,
                        transform=SSDAugmentation(cfg['min_dim'],
                        MEANS))

dataset_val = VOCDetection(dataset_root, [('2007', 'val')],
                       BaseTransform(300, (104, 117, 123)),
                       VOCAnnotationTransform())
def train() :
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net = build_ssd('train', cfg['min_dim'], cfg['num_classes']).to(0)
    for param in net.vgg.parameters():
        param.requires_grad = False
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)


    # loss counters
    loc_loss = 0
    conf_loss = 0
    epochs = 500
    print('Loading the dataset...')

    epoch_size = len(dataset) // batch_size
    print('Training SSD on:', dataset)

    step_index = 0
    data_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=False)

    data_loader_val = data.DataLoader(dataset_val,batch_size,
                                        num_workers=num_workers,
                                        shuffle=False, collate_fn=detection_collate,
                                        pin_memory=False)
                                    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

    epoch_bar = tqdm(total=epochs,desc='epoch')
    for epoch in range(epochs):
        net.train()
        #ibar = tqdm(total=len(data_loader),desc='iteration')
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(0, non_blocking=True)
            targets = [ann.to(0) for ann in targets]

            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            #ibar.write(str(torch.cuda.max_memory_allocated()))
            #ibar.update()
        #ibar.close()

        print('train : ',epoch, loc_loss + conf_loss)
        if epoch % 5 == 0:
            torch.save(net.state_dict(),'/content/drive/My Drive/Colab Notebooks/ssd'+str(epoch)+'.pth')
            eval_func(epoch)
        loc_loss = 0
        conf_loss = 0
        #torch.save(net.module.state_dict(),'ssd.pth')

        
        for i, (images, targets) in enumerate(data_loader_val):
            images = images.to(0, non_blocking=True)
            targets = [ann.to(0) for ann in targets]
        
            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
        
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
        print('val : ',epoch, loc_loss + conf_loss)
        loc_loss = 0
        conf_loss = 0
        scheduler.step()
        epoch_bar.update()
    epoch_bar.close()
if __name__ == '__main__':
    train()
