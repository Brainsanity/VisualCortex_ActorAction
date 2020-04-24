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
from network import net, Ensemble
import time
from utils.eval_metrics import Precision, Recall, F1

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.data_cfg == 'val':
        cfg = val_cfg
    elif args.data_cfg == 'train':
        cfg = train_cfg
    elif args.data_cfg == 'test':
        cfg = test_cfg
    else:
        cfg = val_cfg

    cfg.data_list = args.data_list
    if args.crop != 0:
        cfg.crop_policy='random'        
    if args.fix_norm != None:
        cfg.fix_norm = 1

    if args.net == 'R_2plus1_D' or args.net == 'ensemble':
        test_dataset = a2d_dataset.A2DDataset(cfg, args.dataset_path, is3D=True, nFrames=args.nframes, speed=args.speed)
    else:
        test_dataset = a2d_dataset.A2DDataset(cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    # define load your model here
    if args.net != 'ensemble':
        model = net(43, args.net,args.version).to(device)#
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.load_name + '.ckpt')))
    else:
        model = Ensemble(device, 43, args.pcd_file, args.pcsa_file, args.pcd3d_file, args.pcd_f1, args.pcsa_f1, args.pcd3d_f1)
    
    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))
    print(data_loader.__len__())
    if args.net != 'ensemble':
        model.eval()
    Loss = 0.
    total_step = len(data_loader)
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            if args.net != 'ensemble':
                output = model(images).cpu().detach().numpy()
            else:
                output = model.predict(images,data[2]).cpu().detach().numpy()
            target = labels.cpu().detach().numpy()
            loss = -np.sum( ( target * np.log(output+1e-12) + (1.-target) * np.log(1.-output+1e-12) ).flatten() )
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            X[batch_idx, :] = output
            Y[batch_idx, :] = target

            Loss = Loss + loss
            if batch_idx % 100 == 0:
                print('Step [{}/{}], Loss: {:.4f}'
                      .format(batch_idx, total_step, loss))
        
    P = Precision(X, Y)
    R = Recall(X, Y)
    F = F1(X, Y)
    print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(100 * P, 100 * R, 100 * F))
    print('nImg: {} | loss: {:.4f}'.format( np.sum(X,0), Loss/total_step ) )
    if args.net == 'ensemble':
        print('Evaluation finished for {} on {}'.format( args.pcd_file + ' ' + args.pcsa_file + ' ' + args.pcd3d_file, args.data_list ))
    else:
        print('Evaluation finished for {} on {}'.format( os.path.join(args.model_path, args.load_name + '.ckpt'), args.data_list ))

    if args.note != None:
        f = open( 'Predict_{}_{}_{}.txt'.format( args.data_list, args.net, args.note ), 'w' )
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write('{:.0f} '.format(X[i,j]))
            f.write('\n')
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--pcd_file', type=str, default='models/net.ckpt')
    parser.add_argument('--pcsa_file', type=str, default='models/net.ckpt')
    parser.add_argument('--pcd3d_file', type=str, default='models/net.ckpt')
    parser.add_argument('--pcd_f1', type=float, default=100)
    parser.add_argument('--pcsa_f1', type=float, default=100)
    parser.add_argument('--pcd3d_f1', type=float, default=100)
    parser.add_argument('--load_name', type=str, default='net')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    parser.add_argument('--net', type=str, default='per_class_detection')
    parser.add_argument('--data_list', type=str, default='val')
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--nframes', type=int, default=8)
    parser.add_argument('--speed', type=int, default=2)
    parser.add_argument('--data_cfg', type=str, default='val')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--fix_norm', type=int, default=None, help='fix the normalization bug in a2d_dataset.py')
    args = parser.parse_args()

main(args)
