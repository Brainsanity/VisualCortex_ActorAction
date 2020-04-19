import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    ####
    # define your model
    ####
    def __init__(self, num_classes=43):
        super().__init__()
        self.num_classes = num_classes

        # pretrained faster R-CNN
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        self.pretrained = nn.Sequential(
                                        model.transform,
                                        model.backbone,
                                        model.rpn,
                                        model.roi_heads.box_roi_pool,
                                        )

        # to train
        out_channels = model.bakcbone.out_channels 
        box_roi_pool = MultiScaleRoIAlign(
                                        featmap_names=['0', '1', '2', '3'],
                                        output_size=7,
                                        sampling_ratio=2
                                        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels* resolution ** 2, representation_size)

        linear = nn.Linear(representation_size, num_classes)

        self.totrain = nn.Sequential(
                                    box_roi_pool,
                                    box_head,
                                    linear
                                    )

    def forward(self, input):
        with torch.no_grad():
            features = self.pretrained(input)

        output = self.totrain(features)
        
        return output

