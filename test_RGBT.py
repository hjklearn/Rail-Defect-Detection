import torch as t
from torch import nn
# from RGBT_dataprocessing_CNet import testData1,testData2,testData3
from train_test1.RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from FHENet import Mirror_model
import  numpy as np
from datetime import datetime

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
net = Mirror_model()

net.load_state_dict(t.load('../Pth/FHENet_RGB_D_SOD_rail.pth')) 

a = '../Documents/RGBT-EvaluationTools/SalMap/'
b = 'Net_SOD_rail'
c = ''
path = a + b + c

path1 = path
isExist = os.path.exists(path1)
if not isExist:
	os.makedirs(path1)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader1):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		
		out1 = net(image, depth)
		out = torch.sigmoid(out1[0])

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()

		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')





