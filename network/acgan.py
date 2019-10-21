import torch
import torch.nn as nn


def get_norm(use_sn):
	if use_sn:  # spectral normalization
		return nn.utils.spectral_norm
	else:  # identity mapping
		return lambda x: x

# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def weights_init(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()

# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/utils.py
def weights_init_3channel(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def onehot(y, class_num):
	eye = torch.eye(class_num).type_as(y)  # ny x ny
	onehot = eye[y.view(-1)].float()  # B -> B x ny
	return onehot


# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
class ACGAN_Toy_Generator(nn.Module):
	def __init__(self, nz=2, nc=2, ny=2, dim=512):
		super().__init__()
		self.class_num = ny
		self.net = nn.Sequential(
			nn.Linear(nz + ny, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, nc),
		)
		weights_init(self)

	def forward(self, x, y):
		y = onehot(y, self.class_num)  # B -> B x ny
		x = torch.cat([x, y], dim=1)  # B x (nz + ny)
		return self.net(x)

# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
class ACGAN_Toy_Discriminator(nn.Module):
	def __init__(self, nc=2, ny=2, dim=512, use_sn=False):
		super().__init__()
		norm = get_norm(use_sn)
		self.net = nn.Sequential(
			nn.Linear(nc, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
		)
		self.out_d = nn.Linear(dim, 1)
		self.out_c = nn.Linear(dim, ny)
		weights_init(self)

	def forward(self, x, y=None, get_feature=False):
		x = self.net(x)
		if get_feature:
			return x
		else:
			return self.out_d(x), self.out_c(x)


# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/ACGAN.py
class ACGAN_MNIST_Generator(nn.Module):
	def __init__(self, nz=100, nc=1, ny=10, image_size=32):
		super().__init__()
		self.class_num = ny
		self.image_size = image_size
		self.fc = nn.Sequential(
			nn.Linear(nz + ny, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128 * (image_size // 4) * (image_size // 4)),
			nn.BatchNorm1d(128 * (image_size // 4) * (image_size // 4)),
			nn.ReLU(),
		)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, nc, 4, 2, 1),
			nn.Tanh(),
		)
		weights_init(self)

	def forward(self, x, y):
		y = onehot(y, self.class_num)  # B -> B x ny
		x = torch.cat([x, y], 1)
		x = self.fc(x)
		x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
		x = self.deconv(x)
		return x

# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/ACGAN.py
class ACGAN_MNIST_Discriminator(nn.Module):
	def __init__(self, nc=1, ny=10, image_size=32, use_sn=False):
		super().__init__()
		self.class_num = ny
		self.image_size = image_size
		norm = get_norm(use_sn)
		self.conv = nn.Sequential(
			norm(nn.Conv2d(nc, 64, 4, 2, 1)),  # # use spectral norm
			nn.LeakyReLU(0.2),
			norm(nn.Conv2d(64, 128, 4, 2, 1)),  # use spectral norm
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
		)
		self.fc = nn.Sequential(
			nn.Linear(128 * (image_size // 4) * (image_size // 4), 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2),
		)
		self.out_d = nn.Linear(1024, 1)
		self.out_c = nn.Linear(1024, self.class_num)
		weights_init(self)

	def forward(self, x, y=None, get_feature=False):
		x = self.conv(x)
		x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
		x = self.fc(x)
		if get_feature:
			return x
		else:
			return self.out_d(x), self.out_c(x)


# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class ACGAN_CIFAR10_Generator(nn.Module):
	def __init__(self, nz=100, nc=3, ny=10):
		super().__init__()
		self.class_num = ny
		self.fc = nn.Linear(nz + ny, 384)
		self.tconv = nn.Sequential(
			# tconv1
			nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
			nn.BatchNorm2d(192),
			nn.ReLU(True),
			# tconv2
			nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
			nn.BatchNorm2d(96),
			nn.ReLU(True),
			# tconv3
			nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
			nn.BatchNorm2d(48),
			nn.ReLU(True),
			# tconv4
			nn.ConvTranspose2d(48, nc, 4, 2, 1, bias=False),
			nn.Tanh(),
		)
		weights_init_3channel(self)

	def forward(self, x, y):
		y = onehot(y, self.class_num)  # B -> B x ny
		x = torch.cat([x, y], dim=1)  # B x (nz + ny)
		x = self.fc(x)
		x = x.view(-1, 384, 1, 1)
		x = self.tconv(x)
		return x

# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class ACGAN_CIFAR10_Discriminator(nn.Module):
	def __init__(self, nc=3, ny=10, use_sn=False):
		super().__init__()
		norm = get_norm(use_sn)
		self.conv = nn.Sequential(
			# conv1
			norm(nn.Conv2d(nc, 16, 3, 2, 1, bias=False)),  # use spectral norm
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
			# conv2
			norm(nn.Conv2d(16, 32, 3, 1, 1, bias=False)),  # use spectral norm
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
			# conv3
			norm(nn.Conv2d(32, 64, 3, 2, 1, bias=False)),  # use spectral norm
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
			# conv4
			norm(nn.Conv2d(64, 128, 3, 1, 1, bias=False)),  # use spectral norm
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
			# conv5
			norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)),  # use spectral norm
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
			# conv6
			norm(nn.Conv2d(256, 512, 3, 1, 1, bias=False)),  # use spectral norm
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5, inplace=False),
		)
		self.out_d = nn.Linear(4 * 4 * 512, 1)
		self.out_c = nn.Linear(4 * 4 * 512, ny)
		weights_init_3channel(self)

	def forward(self, x, y=None, get_feature=False):
		x = self.conv(x)
		x = x.view(-1, 4*4*512)
		if get_feature:
			return x
		else:
			return self.out_d(x), self.out_c(x)


