import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
import torch.autograd as autograd
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch.autograd import Variable
import math
import pdb

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

        image_mean = [0.495, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        max_size = 1333
        min_size = 800
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets=None):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        with torch.no_grad():
            features = self.backbone(images.tensors)
            proposals, loss = self.rpn(images, features, targets)

        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.box_head(box_features)
        roi_detections = self.linear(box_features)
        roi_detections = self.transform.postprocess(roi_detections, images.image_sizes, original_image_sizes)

        roi_detections = nn.functional.softmax(roi_detections, 1)
        total_detections = torch.sum(roi_detections, 0) / roi_detections.shape[0]

        return total_detections

