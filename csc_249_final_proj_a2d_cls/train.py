from loader import a2d_dataset
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
import time
import cv2

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.crop != 0:
        train_cfg.crop_policy='random'

    if args.net == 'R_2plus1_D':
        train_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path, is3D=True, nFrames=args.nframes, speed=args.speed)
    else:
        train_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4) # you can make changes

    eval_dataset = a2d_dataset.A2DDataset(val_cfg, args.dataset_path)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define model, Loss, and optimizer
    num_cls = 43
    model = net(num_cls,args.net,args.version).to(device)
    if args.cont != 0:
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'net.ckpt')))
    # criterion = nn.CrossEntropyLoss()
    if args.net == '2_attention_map':
        optimizer = torch.optim.SGD( list(model.base.parameters()) + list(model.top.parameters()) + list(model.attention.parameters()) + list(model.fc_obj.parameters()) + list(model.fc_bgd.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    if args.net == 'per_class_detection':
        optimizer = torch.optim.SGD( list(model.base.parameters()) + list(model.top.parameters()) + list(model.fc.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    if args.net == 'per_class_soft_attention' or args.net == 'per_class_hard_attention':
        optimizer = torch.optim.SGD( list(model.base.parameters()) + list(model.top.parameters()) + list(model.attention.parameters()) + [model.fc_w] + [model.fc_b], lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    if args.net == 'fpn':
        optimizer = torch.optim.SGD( list(model.box_roi_pool.parameters()) + list(model.box_head.parameters()) + list(model.linear.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    if args.net == 'R_2plus1_D':
        optimizer = torch.optim.SGD( list(model.base.parameters()) + list(model.top.parameters()) + list(model.fc.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )

    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        for i, data in enumerate(train_loader):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # Forward, backward and optimize
            outputs = model(images)
            loss = -torch.sum( ( labels * torch.log(outputs) + (1.-labels) * torch.log(1.-outputs) ).flatten() )
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net.ckpt'))
        t2 = time.time()
        print(t2 - t1)
        evaluation(model,eval_loader)

    torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net.ckpt'))

def evaluation(model, eval_loader):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--net', type=str, default='per_class_detection')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--nframes', type=int, default=8)
    parser.add_argument('--speed', type=int, default=2)
    parser.add_argument('--cont', type=int, default=0)  # whether continue the training based on a former net.ckpt
    parser.add_argument('--crop', type=int, default=0)
    args = parser.parse_args()
    print(args)
main(args)
