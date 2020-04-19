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
	def __init__(self, num_classes, name='per_class_detection'):
		super(net, self).__init__()
		self.name = name
		self.num_classes = num_classes
		
		resnet = models.resnet152(pretrained=True)
		self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
		self.top = nn.Sequential(resnet.layer4)

		if name == '2_attention_map':
			self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, 1, kernel_size=3, stride=1, padding=1 )

			self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
			self.fc_obj = nn.Linear( resnet.fc.in_features, num_classes )
			self.fc_bgd = nn.Linear( resnet.fc.in_features, num_classes )

		
		## Per Class Detection Network (PCDN)
		if name == 'per_class_detection':
			# This model converts the multi-object classification problem into a detection problem for each class:
			#   Theoretically we could build a model to perform detection (or classification for pos./neg.) for each class independently,
			# however, to save resources, we use the same feature vector extracted from a deep CNN, and then perform detection/classficication for each class based on that same feature vector
			self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
			self.fc = nn.Linear( resnet.fc.in_features, num_classes*2 )


		## Per Class Detection with Soft/Hard Attention Network (PCDAN: PCDSAN/PCDHAN)
		if name == 'per_class_soft_attention' or name == 'per_class_hard_attention':
			# Based on PCDN, this model introduces attention maps for each class
			#   For each class (including a backgound class), an attention map will be generated based on a feature map extracted from a base CNN,
			# then the feature map will be weighted by the attention map and fed to following layers for per-class detections
			self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, num_classes+1, kernel_size=3, stride=1, padding=1 )
			self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

			## v1
			# self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes*2).cuda() )
			# self.fc_b = torch.nn.Parameter( torch.zeros(num_classes*2).cuda() )
			
			## v2
			self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes).cuda() )
			self.fc_b = torch.nn.Parameter( torch.zeros(num_classes).cuda() )


	def forward(self, images):
		# with torch.no_grad():
		# 	base_feat = self.base(images)
		base_feat = self.base(images)

		if self.name == '2_attention_map':
			atn = self.attention(base_feat)
			atn = torch.softmax(atn.flatten(), 0).reshape(atn.size())
			pooled_feat_obj = self.avgpool(self.top(atn*base_feat))
			pooled_feat_bgd = self.avgpool(self.top((1.-atn)*base_feat))
			scores_obj = self.fc_obj( pooled_feat_obj.view(pooled_feat_obj.size(0),-1) )
			scores_bgd = self.fc_bgd( pooled_feat_bgd.view(pooled_feat_obj.size(0),-1) )
			p_obj = torch.softmax( torch.stack( (scores_obj,scores_bgd), 2 ), 2 )[:,:,0]
			outputs = p_obj


		## Per Class Detection Network (PCDN)
		if self.name == 'per_class_detection':
			outputs = torch.softmax( self.fc( self.avgpool( self.top(base_feat) ).view(base_feat.size(0),-1) ).reshape(base_feat.shape[0],self.num_classes,2), 2 )[:,:,0]


		## Per Class Detection with Soft/Hard Attention Network (PCDAN: PCDSAN/PCDHAN)
		if self.name == 'per_class_soft_attention' or self.name == 'per_class_hard_attention':
			atns = torch.softmax( self.attention(base_feat), 1 )
			if self.name == 'per_class_hard_attention':
				tmp = atns * 0.
				tmp[ atns == torch.max(atns,1)[0].unsqueeze(1) ] = 1.
				atns = tmp
			
			## v1
			# outputs = torch.softmax( torch.stack( tuple([ torch.sum( self.avgpool( self.top( base_feat * atns[:,i,:,:].unsqueeze(1) ) ).view(base_feat.size(0),-1,1) * self.fc_w[:,2*i:2*(i+1)].unsqueeze(0), 1 ) + self.fc_b[i*2:(i+1)*2].unsqueeze(0) for i in range(self.num_classes) ]), 2 ), 1 )[:,0,:]
			
			## v2
			outputs = torch.sigmoid( torch.stack( tuple([ torch.sum( self.avgpool( self.top( base_feat * atns[:,i,:,:].unsqueeze(1) ) ).view(base_feat.size(0),-1) * self.fc_w[:,i], 1 ) + self.fc_b[i] for i in range(self.num_classes) ]), 1 ) )

		return outputs


