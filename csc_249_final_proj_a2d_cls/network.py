import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    def __init__(self, num_classes):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(net, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.linear1 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.linear2 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.linear3 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.bn1 = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.bn2 = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.bn3 = nn.BatchNorm1d(num_classes, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        # with torch.no_grad():
        #     features = self.resnet(images)
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        features = self.linear(features)
        # features = torch.softmax( self.bn1(self.linear1(features)), 1 ) + torch.softmax( self.bn2(self.linear2(features)), 1 ) + torch.softmax( self.bn3(self.linear3(features)), 1 )
        return features
  
