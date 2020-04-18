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
from network import net
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4) # you can make changes

    # Define model, Loss, and optimizer
    # model = ###
    # criterion = ###
    # optimizer = ###
    model = net(43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( list(model.base.parameters()) + list(model.top.parameters()) + list(model.attention.parameters()) + list(model.fc_obj.parameters()) + list(model.fc_bgd.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    # optimizer = torch.optim.SGD( list(model.linear.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    # optimizer = torch.optim.SGD( list(model.linear.parameters()) + list(model.bn.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    # optimizer = torch.optim.SGD( list(model.resnet.parameters()) + list(model.linear.parameters()) + list(model.bn.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )
    # optimizer = torch.optim.SGD( list(model.resnet.parameters()) + list(model.linear1.parameters()) + list(model.linear2.parameters()) + list(model.linear3.parameters()) + list(model.bn1.parameters()) + list(model.bn2.parameters()) + list(model.bn3.parameters()), lr=0.00001, momentum=train_cfg.optimizer['args']['momentum'], dampening=0, weight_decay=train_cfg.optimizer['args']['weight_decay'], nesterov=False )

    # Train the models
    # STA_imgs = torch.zeros(43,3,train_cfg.crop_size[0],train_cfg.crop_size[1]).to(device)
    nImages = 0.
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        for i, data in enumerate(data_loader):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # for iImg in range(images.shape[0]):
            #     STA_imgs[labels[iImg,:].long(),:,:,:] = STA_imgs[labels[iImg].long(),:,:,:] + 1.
            #     nImages = nImages+1.

            # Forward, backward and optimize
            outputs = model(images)
            # loss = criterion(outputs, labels)
            # loss = - torch.sum( torch.sum( ( nn.LogSoftmax(1)(outputs) * labels ), 1 ) / torch.max( torch.stack( ( torch.sum(labels,1), torch.ones(labels.shape[0]).to(device) ) ), 0 )[0] )
            # loss = torch.sum( (outputs - labels).flatten()**2 ) / torch.sum(labels.flatten())
            loss = -torch.sum( ( labels * torch.log(outputs) + (1.-labels) * torch.log(1.-outputs) ).flatten() )
            # loss = - torch.sum( torch.sum( ( torch.log(outputs) * labels ), 1 ) / torch.max( torch.stack( ( torch.sum(labels,1), torch.ones(labels.shape[0]).to(device) ) ), 0 )[0] )
            # loss = - torch.sum( torch.sum( ( torch.log(outputs) * labels ), 1 ) )
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
        print(outputs)
        # STA_imgs = STA_imgs / nImages
        # for i in range(43):
        #     torchvision.utils.save_image( STA_imgs[i,:,:,:].squeeze(0), 'sta{}.png'.format(i) )
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
    args = parser.parse_args()
    print(args)
main(args)
