import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from GCN import GGCN
from Xception import xception

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

class early_fusion(nn.Module):
	def __init__(self):
		super(early_fusion,self).__init__()
		self.Linearlayer = nn.Linear(224*224,256)
	def forward(self,rgbfuture, skeleton_future):
		result = []
		np.set_printoptions(threshold=np.inf)
		for i in range(skeleton_future.size(0)):
			mask_code = []
			for j in range(skeleton_future[i].size(0)):
				if j == len(skeleton_future[0])//2:
					continue
				J_middle = skeleton_future[i][len(skeleton_future[0])//2]
				distances = np.linalg.norm(skeleton_future[i][j].cpu() - J_middle.cpu(), axis=1)
				j_max = np.argmax(distances)
				W, H = 224, 224
				M_ske = np.zeros((1, W, H))

				pixel_x, pixel_y, _ = skeleton_future[i][j][j_max]*224
				pixel_x, pixel_y = int(pixel_x),int(pixel_y)

				if pixel_y<=204 and pixel_y>=20:
					M_ske[0, pixel_x-20:pixel_x+20, pixel_y-20:pixel_y+20] = 1
				elif pixel_y<20:
					M_ske[0, pixel_x-20:pixel_x+20, 0:pixel_y+20] = 1
				elif pixel_y>204:
					M_ske[0, pixel_x-20:pixel_x+20, pixel_y-20:224] = 1
				mask_code.append(M_ske)
			mask_code = torch.Tensor(mask_code)
			merged_tensor = torch.max(mask_code, dim=0)[0].to(device)
			merged_tensor = merged_tensor.view(-1)
			merged_tensor = self.Linearlayer(merged_tensor)
			result.append(rgbfuture[i]*merged_tensor)
		result = torch.stack(result).to(device)	
		return result
class Model(nn.Module):
	def __init__(self, adj, num_v, num_classes_g, num_classes_x, gc_dims, sc_dims, feat_dims, dropout=0.5):
		super(Model, self).__init__()
		self.gcn = GGCN(adj, num_v, num_classes_g, gc_dims, sc_dims, feat_dims)
		self.xception = xception(pretrained=True)
		self.early_fusion = early_fusion()
		self.early_fusion_linearlayer = nn.Linear(num_classes_g*2,num_classes_g)
		self.last_fusion_linearlayer = nn.Linear(num_classes_g*2,5)
	def forward(self, image, skeleton):
		
		skeleton_future = self.gcn(skeleton)
		
		rgbfuture = self.xception(image)
		early_fusion = self.early_fusion(rgbfuture,skeleton)
		
		rgbstream_concat = torch.cat((rgbfuture,early_fusion),dim = 1)
		rgbstream_concat = self.early_fusion_linearlayer(rgbstream_concat)
		result = torch.cat((skeleton_future,rgbfuture),dim = 1)
		result = self.last_fusion_linearlayer(result)
		return result
		
