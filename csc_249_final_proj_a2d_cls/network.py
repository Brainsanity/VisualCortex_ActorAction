import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.image_list import ImageList
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

        self.backbone = model.backbone
        self.rpn = model.rpn
        self.rpn.training = False

        # to train
        out_channels = model.backbone.out_channels
        self.box_roi_pool = MultiScaleRoIAlign(
                                        featmap_names=['0', '1', '2', '3'],
                                        output_size=7,
                                        sampling_ratio=2
                                        )

        resolution = self.box_roi_pool.output_size[0]
        representation_size = 1024
        self.box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        self.linear = nn.Linear(representation_size, num_classes)

    def forward(self, images):
        image_shapes = [x.shape[1:] for x in images]
        image_list = ImageList(images, image_shapes)

        with torch.no_grad():
            features = self.backbone(images)
            proposals, loss = self.rpn(image_list, features)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        output = self.linear(box_features)

        return output

