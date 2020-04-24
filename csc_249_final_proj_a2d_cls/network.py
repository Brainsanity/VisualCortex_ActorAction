import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math
# from torchvision.ops import MultiScaleRoIAlign
# from torchvision.models.detection.image_list import ImageList
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

class net(nn.Module):
	def __init__(self, num_classes, name='per_class_detection', version=None):
		super(net, self).__init__()
		self.name = name
		self.num_classes = num_classes
		self.version = version

		if name == '2_attention_map':
			resnet = models.resnet152(pretrained=True)
			self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
			self.top = nn.Sequential(resnet.layer4)
			self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, 1, kernel_size=3, stride=1, padding=1 )

			self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
			self.fc_obj = nn.Linear( resnet.fc.in_features, num_classes )
			self.fc_bgd = nn.Linear( resnet.fc.in_features, num_classes )

		
		## Per Class Detection Network (PCDN)
		if name == 'per_class_detection':
			# This model converts the multi-object classification problem into a detection problem for each class:
			#   Theoretically we could build a model to perform detection (or classification for pos./neg.) for each class independently,
			# however, to save resources, we use the same feature vector extracted from a deep CNN, and then perform detection/classficication for each class based on that same feature vector
			if self.version == None:
				self.version = '1'

			## v1
			if self.version == '1':
				resnet = models.resnet152(pretrained=True)
				self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
				self.top = nn.Sequential(resnet.layer4)
				self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
				self.fc = nn.Linear( resnet.fc.in_features, num_classes*2 )
				# self.bn = nn.BatchNorm1d( num_classes*2, momentum=0.01 )		# much worse with this!!!

			## v2
			if self.version == '2':
				resnet = models.resnet152(pretrained=True)
				self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
				self.top = nn.Sequential(resnet.layer4)
				self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
				self.fc = nn.Linear( resnet.fc.in_features, num_classes )


		## Per Class Detection with Soft/Hard Attention Network (PCDAN: PCDSAN/PCDHAN)
		if name == 'per_class_soft_attention' or name == 'per_class_hard_attention':
			# Based on PCDN, this model introduces attention maps for each class
			#   For each class (including a backgound class), an attention map will be generated based on a feature map extracted from a base CNN,
			# then the feature map will be weighted by the attention map and fed to following layers for per-class detections
			resnet = models.resnet152(pretrained=True)
			self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
			self.top = nn.Sequential(resnet.layer4)
			self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, num_classes+1, kernel_size=3, stride=1, padding=1 )
			self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

			## v1
			if self.version == '1':
				self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes*2).cuda() )
				self.fc_b = torch.nn.Parameter( torch.zeros(num_classes*2).cuda() )
			
			## v2
			if self.version == '2':
				self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes).cuda() )
				self.fc_b = torch.nn.Parameter( torch.zeros(num_classes).cuda() )

			## v3
			if self.version == '3':
				self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes+1).cuda() )
				self.fc_b = torch.nn.Parameter( torch.zeros(num_classes+1).cuda() )

			## v4
			if self.version == '4':
				self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, 1, kernel_size=3, stride=1, padding=1 )
				self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes).cuda() )
				self.fc_b = torch.nn.Parameter( torch.zeros(num_classes).cuda() )

			## v5
				self.attention = nn.Conv2d( resnet.layer3[-1].conv3.out_channels, 1, kernel_size=3, stride=1, padding=1 )
				self.fc_w = torch.nn.Parameter( torch.zeros(resnet.fc.in_features, num_classes*2).cuda() )
				self.fc_b = torch.nn.Parameter( torch.zeros(num_classes*2).cuda() )

		## Faster FPN
		if name == 'fpn':
			# pretrained faster R-CNN
			model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

			self.backbone = model.backbone
			self.backbone.train = False
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
			self.linear = nn.Linear(representation_size, num_classes+1)


		## 3D Per Class Detection Network (PCDN3D)
		if name == 'R_2plus1_D':
			model = models.video.r2plus1d_18(pretrained=True,progress=False)
			self.base = nn.Sequential(model.stem, model.layer1, model.layer2, model.layer3)
			self.top = nn.Sequential(model.layer4)
			self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
			self.fc = nn.Linear( model.fc.in_features, num_classes*2 )



	def forward(self, images):

		if self.name == '2_attention_map':
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
			outputs = p_obj


		## Per Class Detection Network (PCDN)
		if self.name == 'per_class_detection':
			# with torch.no_grad():
			# 	base_feat = self.base(images)
			base_feat = self.base(images)

			## v1
			if self.version == '1':
				outputs = torch.softmax( self.fc( self.avgpool( self.top(base_feat) ).view(base_feat.size(0),-1) ).reshape(base_feat.shape[0],self.num_classes,2), 2 )[:,:,0]

			if self.version == '2':
				outputs = torch.sigmoid( self.fc( self.avgpool( self.top(base_feat) ).view(base_feat.size(0),-1) ) )


		## Per Class Detection with Soft/Hard Attention Network (PCDAN: PCDSAN/PCDHAN)
		if self.name == 'per_class_soft_attention' or self.name == 'per_class_hard_attention':
			# with torch.no_grad():
			# 	base_feat = self.base(images)
			base_feat = self.base(images)

			if self.version != '4':
				atns = torch.softmax( self.attention(base_feat), 1 )
				if self.name == 'per_class_hard_attention':
					tmp = atns * 0.
					tmp[ atns == torch.max(atns,1)[0].unsqueeze(1) ] = 1.
					atns = tmp
			
			## v1
			if self.version == '1':
				outputs = torch.softmax( torch.stack( tuple([ torch.sum( self.avgpool( self.top( base_feat * atns[:,i,:,:].unsqueeze(1) ) ).view(base_feat.size(0),-1,1) * self.fc_w[:,2*i:2*(i+1)].unsqueeze(0), 1 ) + self.fc_b[i*2:(i+1)*2].unsqueeze(0) for i in range(self.num_classes) ]), 2 ), 1 )[:,0,:]
			
			## v2
			if self.version == '2':
				outputs = torch.sigmoid( torch.stack( tuple([ torch.sum( self.avgpool( self.top( base_feat * atns[:,i,:,:].unsqueeze(1) ) ).view(base_feat.size(0),-1) * self.fc_w[:,i], 1 ) + self.fc_b[i] for i in range(self.num_classes) ]), 1 ) )

			## v3
			# add bach normalization???
			if self.version == '3':
				outputs = torch.softmax( torch.stack( tuple([ torch.sum( self.avgpool( self.top( base_feat * atns[:,i,:,:].unsqueeze(1) ) ).view(base_feat.size(0),-1) * self.fc_w[:,i], 1 ) + self.fc_b[i] for i in range(self.num_classes+1) ]), 1 ), 1 )
				outputs = outputs[:,:-1] / (outputs[:,:-1] + outputs[:,-1].unsqueeze(1) + 1e-15)

			## v4
			if self.version == '4':
				atns = torch.sigmoid( self.attention(base_feat) )
				outputs = torch.sigmoid( torch.sum( self.avgpool( self.top(base_feat*atns) ).view(base_feat.size(0),-1,1) * self.fc_w, 1 ) + self.fc_b )

			## v5
			if self.version == '5':
				atns = torch.sigmoid( self.attention(base_feat) )
				outputs = torch.softmax( ( torch.sum( self.avgpool( self.top(base_feat*atns) ).view(base_feat.size(0),-1,1) * self.fc_w, 1 ) + self.fc_b ).reshape(images.shape[0],self.num_classes,2), 2 )[:,:,0]


		## fpn
		if self.name == 'fpn':
			image_shapes = [x.shape[1:] for x in images]
			image_list = ImageList(images, image_shapes)		
			
			with torch.no_grad():
				features = self.backbone(images)
				proposals, loss = self.rpn(image_list, features)			
			
			box_features = self.box_roi_pool(features, proposals, image_shapes)
			box_features = self.box_head(box_features)
			# scores = torch.softmax( self.linear(box_features), 1 )[:-1]

			# idx = 0
			# outputs = []
			# for i in range(len(proposals)):
			# 	outputs.append( torch.sum( scores[ idx : proposals[i].shape[0], : ], 0 ) / proposals[i].shape[0] )
			# 	idx += proposals[i].shape[0]
			# outputs = torch.stack(outputs, 0)

			roi_detections = self.linear(box_features)
			roi_detections = self.transform.postprocess(roi_detections, images.image_sizes, original_image_sizes)
			
			roi_detections = nn.functional.softmax(roi_detections, 1)
			outputs = torch.sum(roi_detections, 0) / roi_detections.shape[0]


		if self.name == 'R_2plus1_D':
			base_feat = self.base(images)
			outputs = torch.softmax( self.fc( self.avgpool( self.top(base_feat) ).view(base_feat.size(0),-1) ).reshape(base_feat.shape[0],self.num_classes,2), 2 )[:,:,0]


		return outputs



class Ensemble():
	# Weighted (by performance) Voting + Weighted (by performance) Averaging
	# e.g.: given 0.2, 0.6, 0.61
	# 		voting:		2/3 = 0.67				=> 1		(this one does not take into account the confidence level of each vote)
	#		averaging:	1.41/3 = 0.47			=> 0		(this one is heavily affected by extreme value)
	#		combined:	(0.67+0.47)/2 = 0.57	=> 1		(balance between the other two)

	def __init__(self, pcd_file, pcsa_file, pcd3d_file):
		netPCD = net( 'per_class_detection' )
		netPCSA = net( 'per_class_soft_attention', '4' )


	def predict(self, images, frames):
		pass