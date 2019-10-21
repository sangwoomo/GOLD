# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
import torch
import torch.nn as nn


class GANLoss(nn.Module):
	def __init__(self, target_real_label=1.0, target_fake_label=0.0, reduction='mean'):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

	def get_target_tensor(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def __call__(self, prediction, target_is_real):
		target_tensor = self.get_target_tensor(prediction, target_is_real)
		loss = self.loss(prediction, target_tensor)
		return loss


