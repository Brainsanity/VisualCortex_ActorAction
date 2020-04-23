import os
import sys
sys.path.append('')
import random
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import loader.transforms as tf
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from torchvision import transforms

def to_cls(image_label, num_class):
    '''

    :param image_label:
    :return: label encoding for multi-label multi-class training
    '''
    image_label.flatten()
    label = np.zeros(num_class)
    for i in range(num_class):
        if i in image_label:
            label[i] = 1
    return label

class A2DDataset(Dataset):

    #num_class = 43
    num_class_orig = 80
    ignore_label = 255
    background_label = 0

    # 35+8=43 valid classes
    valid_cls = [
        11, 12, 13, 15, 16, 17, 18, 19,    # 1-8
        21, 22, 26, 28, 29,    # 9-13
        34, 35, 36, 39,    # 14-17
        41, 43, 44, 45, 46, 48, 49,    # 18-24
        54, 55, 56, 57, 59,    # 25-29
        61, 63, 65, 66, 67, 68, 69,    # 30-36
        72, 73, 75, 76, 77, 78, 79] # 37-43
    num_valid_cls = len(valid_cls)
    convert_label = dict()
    convert_label_back = dict()
    for i, label in enumerate(valid_cls):
        convert_label[label] = i
        convert_label_back[i] = label

    label_80to43 = np.ones((num_class_orig))*255
    for label in range(num_class_orig):
        if label in convert_label:
            label_80to43[label] = convert_label[label]
    label_80to43 = label_80to43.astype(np.uint8)
    #print(label_80to43)
    label_43to80 = np.ones((num_valid_cls))*255
    for i in range(num_valid_cls):
        label_43to80[i] = convert_label_back[i]
    label_43to80 = label_43to80.astype(np.uint8)

    # official color map
    cmap = np.array([[0,0,0],   # 0
        #[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [52,1,1],[103,1,1],[154,1,1],[205,1,1],[255,1,1],   # 11-15
        [255,51,51],[255,103,103],[255,154,154],[255,205,205],[0,0,0],  # 16-20
        [52,46,1],[103,92,1],[154,138,1],[205,184,1],[255,230,1],   # 21-25
        [255,235,51],[255,240,103],[255,245,154],[255,250,205],[0,0,0], # 26-30
        [11,52,1],[21,103,1],[31,154,1],[41,205,1],[52,255,1],  # 31-35
        [92,255,51],[133,255,103],[174,255,154],[215,255,205],[0,0,0],  # 36-40
        [1,52,36],[1,103,72],[1,154,108],[1,205,143],[1,255,179],   #41-45
        [51,255,194],[103,255,210],[154,255,225],[205,255,240],[0,0,0], # 46-50
        [1,21,52],[1,41,103],[1,62,154],[1,82,205],[1,103,255], # 51-55
        [51,133,255],[103,164,255],[154,194,255],[205,225,255],[0,0,0], # 56-60
        [26,1,52],[52,1,103],[77,1,154],[103,1,205],[128,1,255],    # 61-65
        [154,51,255],[179,103,255],[205,154,255],[230,205,255],[0,0,0], #66-70
        [52,1,31],[103,1,62],[154,1,92],[205,1,123],[255,1,153],    #71-75
        [255,51,174],[255,103,194],[255,154,215],[255,205,235]  #76-79
    ])
    cmap = cmap.astype(np.uint8)

    # 80=1(bg) + 9+7(none) + 7(actor)*9(action)
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
    def __init__(self, config, dataset_path, is3D=False, nFrames=8, speed=2):
        # speed: for 3D version, extract one frame for each speed frames
        super(A2DDataset, self).__init__()
      
        with open(
                os.path.join(dataset_path, 'list',
                             config.data_list + '.txt')) as f:
            self.img_list = []
            for line in f:
                if line[-1] == '\n':
                    self.img_list.append(line[:-1])
                else:
                    self.img_list.append(line)

        self.img_dir = os.path.join(dataset_path, 'pngs320H')
        self.gt_dir = os.path.join(dataset_path, 'Annotations/mat')
        self.config = config
        self.class_names = [A2DDataset.class_names[cls] for cls in A2DDataset.valid_cls]
        self.is3D = is3D
        self.nFrames = 1

        if is3D:
            self.img_nFrames = []
            with open( os.path.join(dataset_path, 'list', config.data_list + '_nFrames.txt') ) as f:
                self.img_nFrames = []
                for line in f:
                    if line[-1] == '\n':
                        self.img_nFrames.append(int(line[:-1]))
                    else:
                        self.img_nFrames.append(int(line))
            self.nFrames = nFrames
            self.speed = speed

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        clipName = self.img_list[idx][:-5]
        iFrame = int(self.img_list[idx][-5:])

        if self.is3D:
            maxFrame = self.img_nFrames[idx]
            iFrames = ( np.arange(self.nFrames).astype(int) - np.floor((self.nFrames-1)/2).astype(int) ) * self.speed + iFrame
            if iFrames[0] < 1:
                iFrames = iFrames - iFrames[0] + 1
            if iFrames[-1] > maxFrame:
                iFrames = iFrames - iFrames[-1] + maxFrame
        else:
            iFrames = np.ones(1,int)*iFrame

        image_label = list()
        cv2_INTERP = list()
        input_mean = list()
        for i in range(len(iFrames)):
            if hasattr(self.config, 'fix_norm'):
                img = cv2.imread( os.path.join(self.img_dir, clipName, '{:05d}.png'.format(iFrames[i])) )
            else:
                img = cv2.imread( os.path.join(self.img_dir, clipName, '{:05d}.png'.format(iFrames[i])) ).astype(np.float32)
            image_label.append(img)
            cv2_INTERP.append(cv2.INTER_LINEAR)
            input_mean.append(self.config.input_mean)
        cv2_INTERP.append(cv2.INTER_NEAREST)
        input_mean.append(A2DDataset.background_label)

        gt_load_path = os.path.join(self.gt_dir, clipName + '{:05d}.mat'.format(iFrame))
        label_orig = h5py.File(gt_load_path,'r')['reS_id']
        label_orig = np.transpose(label_orig)
        label = label_orig #A2DDataset.label_80to43[label_orig]

        image_label.append(label)

        # flip
        if hasattr(self.config, 'flip') and self.config.flip:
            image_label = tf.group_random_flip(image_label)

        if hasattr(self.config, 'crop_policy'):
            target_size = self.config.crop_size
            if self.config.crop_policy == 'none':
                # resize
                image_label = tf.group_rescale(image_label,
                                                #self.config.scale_factor,
                                                target_size,
                                                cv2_INTERP)
            else:
                # resize -> crop -> pad 
                image_label = tf.group_rescale(image_label,
                                                self.config.scale_factor,
                                                cv2_INTERP)
                if self.config.crop_policy == 'random':
                    image_label = tf.group_random_crop(image_label, target_size)
                    image_label = tf.group_random_pad(
                        image_label, target_size,
                        input_mean)
                elif self.config.crop_policy == 'center':
                    image_label = tf.group_center_crop(image_label, target_size)
                    image_label = tf.group_concer_pad(
                        image_label, target_size,
                        input_mean)
                else:
                    ValueError('Unknown crop policy: {}'.format(
                        self.config.crop_policy))
        if hasattr(self.config, 'rotation') and random.random() < 0.5:
            image_label = tf.group_rotation(
                image_label, self.config.rotation,
                cv2_INTERP,
                input_mean)
        # blur
        if hasattr(self.config,
                   'blur') and self.config.blur and random.random() < 0.5:
            for i in range(len(iFrames)):
                image_label[i] = tf.blur(image_label[i])

        for i in range(len(iFrames)):
            image_label[i] = cv2.resize(image_label[i], (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        for i in range(len(iFrames)):
            image_label[i] = transform(image_label[i])
            image_label[i] = image_label[i].contiguous().float()
        if self.is3D:
            image = torch.stack( image_label[:-1], 1 )
        else:
            image = image_label[0]
        label = A2DDataset.label_80to43[image_label[-1]]
        label = to_cls(label, 43)
        label = torch.from_numpy(label).contiguous().long()
        return image, label

class A2DDataset_test(Dataset):

    num_class_orig = 80
    ignore_label = 255
    background_label = 0

    # 35+8=43 valid classes
    valid_cls = [
        11, 12, 13, 15, 16, 17, 18, 19,    # 1-8
        21, 22, 26, 28, 29,    # 9-13
        34, 35, 36, 39,    # 14-17
        41, 43, 44, 45, 46, 48, 49,    # 18-24
        54, 55, 56, 57, 59,    # 25-29
        61, 63, 65, 66, 67, 68, 69,    # 30-36
        72, 73, 75, 76, 77, 78, 79] # 37-43
    num_valid_cls = len(valid_cls)
    convert_label = dict()
    convert_label_back = dict()
    for i, label in enumerate(valid_cls):
        convert_label[label] = i
        convert_label_back[i] = label

    label_80to43 = np.ones((num_class_orig))*255
    for label in range(num_class_orig):
        if label in convert_label:
            label_80to43[label] = convert_label[label]
    label_80to43 = label_80to43.astype(np.uint8)
    #print(label_80to43)
    label_43to80 = np.ones((num_valid_cls))*255
    for i in range(num_valid_cls):
        label_43to80[i] = convert_label_back[i]
    label_43to80 = label_43to80.astype(np.uint8)

    # official color map
    cmap = np.array([[0,0,0],   # 0
        #[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [52,1,1],[103,1,1],[154,1,1],[205,1,1],[255,1,1],   # 11-15
        [255,51,51],[255,103,103],[255,154,154],[255,205,205],[0,0,0],  # 16-20
        [52,46,1],[103,92,1],[154,138,1],[205,184,1],[255,230,1],   # 21-25
        [255,235,51],[255,240,103],[255,245,154],[255,250,205],[0,0,0], # 26-30
        [11,52,1],[21,103,1],[31,154,1],[41,205,1],[52,255,1],  # 31-35
        [92,255,51],[133,255,103],[174,255,154],[215,255,205],[0,0,0],  # 36-40
        [1,52,36],[1,103,72],[1,154,108],[1,205,143],[1,255,179],   #41-45
        [51,255,194],[103,255,210],[154,255,225],[205,255,240],[0,0,0], # 46-50
        [1,21,52],[1,41,103],[1,62,154],[1,82,205],[1,103,255], # 51-55
        [51,133,255],[103,164,255],[154,194,255],[205,225,255],[0,0,0], # 56-60
        [26,1,52],[52,1,103],[77,1,154],[103,1,205],[128,1,255],    # 61-65
        [154,51,255],[179,103,255],[205,154,255],[230,205,255],[0,0,0], #66-70
        [52,1,31],[103,1,62],[154,1,92],[205,1,123],[255,1,153],    #71-75
        [255,51,174],[255,103,194],[255,154,215],[255,205,235]  #76-79
    ])
    cmap = cmap.astype(np.uint8)

    # 80=1(bg) + 9+7(none) + 7(actor)*9(action)
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
    def __init__(self, config, dataset_path):
        super(A2DDataset_test, self).__init__()
        with open(
                os.path.join(dataset_path,'list',
                             config.data_list + '.txt')) as f:
            self.img_list = []
            for line in f:
                if line[-1] == '\n':
                    self.img_list.append(line[:-1])
                else:
                    self.img_list.append(line)

        self.img_dir = os.path.join(dataset_path, 'pngs320H')
        self.config = config
        self.class_names = [A2DDataset.class_names[cls] for cls in A2DDataset.valid_cls]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        vd_frame_idx = self.img_list[idx]
        image_path = os.path.join(self.img_dir, vd_frame_idx + '.png')
        image = cv2.imread(image_path).astype(np.float32)


        image = cv2.resize(image, (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        image = transform(image)
        image = image.contiguous().float()

        return image

if __name__ == '__main__':

    # train_dataset = A2DDataset(train_cfg)
    # val_dataset = A2DDataset(val_cfg)

    # load training or validation datasets
    train_dataset = A2DDataset(train_cfg, '../A2D')
    dataloader = DataLoader(train_dataset, batch_size=4,
                           shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        print(data[0].size(), data[1].size())
        break

    # load test datasets
    test_dataset = A2DDataset_test(test_cfg, '../A2D')
    dataloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        print(data.size())
        break
