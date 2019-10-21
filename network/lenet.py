# https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_32(nn.Module):
	def __init__(self, n_channels=3, n_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(n_channels, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, n_classes)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out


# Create LeNet classifier
def LeNet(image_size, n_channels=3, n_classes=10):
	if image_size == 32:
		return LeNet_32(n_channels, n_classes)
	else:
		raise NotImplementedError

