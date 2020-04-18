import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net1(nn.Module):
    def __init__(self, num_classes):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(net, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        # modules = list(resnet.children())[:-2]      # delete the last avgpool layer and fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, num_classes)
        # self.linear = nn.Linear(resnet.fc.in_features * resnet.avgpool.kernel_size**2, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.linear1 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.linear2 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.linear3 = nn.Linear(resnet.fc.in_features, num_classes)
        # self.bn1 = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.bn2 = nn.BatchNorm1d(num_classes, momentum=0.01)
        # self.bn3 = nn.BatchNorm1d(num_classes, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        # features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        features = self.linear(features)
        # features = torch.softmax( self.bn1(self.linear1(features)), 1 ) + torch.softmax( self.bn2(self.linear2(features)), 1 ) + torch.softmax( self.bn3(self.linear3(features)), 1 )
        return features
  

class net(nn.Module):
	def __init__(self, num_classes):
		super(net, self).__init__()
		self.num_classes = num_classes
		
		resnet = models.resnet152(pretrained=True)
		self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
		self.top = nn.Sequential(resnet.layer4)

		# self.conv_3_3 = nn.Conv2d( 1024, 512, kernel_size=(3,3), stride=1, padding=1 )
		self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, 1, kernel_size=3, stride=1, padding=1 )

		self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
		self.fc_obj = nn.Linear( resnet.fc.in_features, num_classes )
		self.fc_bgd = nn.Linear( resnet.fc.in_features, num_classes )

	def forward(self, images):
		# with torch.no_grad():
		# 	base_feat = self.base(images)
		base_feat = self.base(images)
		atn = self.attention(base_feat)
		atn = torch.softmax(atn.flatten(), 0).reshape(atn.size())
		pooled_feat_obj = self.avgpool(self.top(atn*base_feat))
		pooled_feat_bgd = self.avgpool(self.top((1.-atn)*base_feat))
		scores_obj = self.fc_obj( pooled_feat_obj.view(pooled_feat_obj.size(0),-1) )
		scores_bgd = self.fc_bgd( pooled_feat_bgd.view(pooled_feat_obj.size(0),-1) )
		p_obj = torch.softmax( torch.stack( (scores_obj,scores_bgd), 2 ), 2 )[:,:,0]

		return p_obj


