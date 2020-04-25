from loader import a2d_dataset
import argparse
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import test as test_cfg
# from network import Res152_MLMC
import pickle

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valid_cls = [
    11, 12, 13, 15, 16, 17, 18, 19,
    21, 22, 26, 28, 29,
    34, 35, 36, 39,
    41, 43, 44, 45, 46, 48, 49,
    54, 55, 56, 57, 59,
    61, 63, 65, 66, 67, 68, 69,
    72, 73, 75, 76, 77, 78, 79]
class_names = np.array([
    'background',
    'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
    'none',
    'adult-climbing',
    'adult-crawling',
    'adult-eating',
    'adult-flying',
    'adult-jumping',
    'adult-rolling',
    'adult-running',
    'adult-walking',
    'adult-none',
    'none',
    'baby-climbing',
    'baby-crawling',
    'baby-eating',
    'baby-flying',
    'baby-jumping',
    'baby-rolling',
    'baby-running',
    'baby-walking',
    'baby-none',
    'none',
    'ball-climbing',
    'ball-crawling',
    'ball-eating',
    'ball-flying',
    'ball-jumping',
    'ball-rolling',
    'ball-running',
    'ball-walking',
    'ball-none',
    'none',
    'bird-climbing',
    'bird-crawling',
    'bird-eating',
    'bird-flying',
    'bird-jumping',
    'bird-rolling',
    'bird-running',
    'bird-walking',
    'bird-none',
    'none',
    'car-climbing',
    'car-crawling',
    'car-eating',
    'car-flying',
    'car-jumping',
    'car-rolling',
    'car-running',
    'car-walking',
    'car-none',
    'none',
    'cat-climbing',
    'cat-crawling',
    'cat-eating',
    'cat-flying',
    'cat-jumping',
    'cat-rolling',
    'cat-running',
    'cat-walking',
    'cat-none',
    'none',
    'dog-climbing',
    'dog-crawling',
    'dog-eating',
    'dog-flying',
    'dog-jumping',
    'dog-rolling',
    'dog-running',
    'dog-walking',
    'dog-none',
])

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_cfg.isTest = True
    test_dataset = a2d_dataset.A2DDataset(test_cfg, args.dataset_path, is3D=True, nFrames=16, speed=1)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # define and load pre-trained model
    if args.net != 'ensemble':
        model = net(43, args.net,args.version).to(device)#
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.load_name + '.ckpt')))
    else:
        model = Ensemble(device, 43, args.pcd_file, args.pcsa_file, args.pcd3d_file, args.pcd_f1, args.pcsa_f1, args.pcd3d_f1)

    results = np.zeros((data_loader.__len__(), args.num_cls))
    model.eval()

    # prediction and saving
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            output = model(images).cpu().detach().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            results[batch_idx, :] = output
    with open('results_netid.pkl', 'wb') as f:
        pickle.dump(results, f)


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
