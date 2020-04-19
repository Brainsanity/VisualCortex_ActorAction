import torch
import torch.nn as nn
import torchvision.models as models
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

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # train the predictor
        model.roi_heads.box_predictor = models.detection.FasterRCNNPredictor(in_features, num_classes)


    def forward(self, input):
        labels, bbox = self.model(input)
        return labels

