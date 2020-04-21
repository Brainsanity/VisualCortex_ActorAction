from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
import cv2

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.data_cfg == 'val':
        cfg = val_cfg
    elif args.data_cfg == 'train':
        cfg = train_cfg
    elif args.data_cfg == 'test':
        cfg = test_cfg
    else:
        cfg = val_cfg
    
    cfg.data_list = args.data_list

    test_dataset = a2d_dataset.A2DDataset(cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    nObjs = torch.zeros(43).cuda()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            nObjs = nObjs + torch.sum(labels,0)
            for i in range(labels.shape[0]):
                if labels[i,0] > 0:
                    pass
                    # cv2.imwrite( '../adult_climing/{}_{}.png'.format(batch_idx,i), ( np.array(images[i,:,:,:].squeeze(0).permute(1,2,0).cpu()) - 20. ) / 1200 * 255 )
            # if batch_idx == 20:
            #     break

    print(nObjs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--data_cfg', type=str, default='val')
    parser.add_argument('--data_list', type=str, default='val')
    args = parser.parse_args()

main(args)
