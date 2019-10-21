import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


"""""""""""""""	
Generate noise
"""""""""""""""

def make_z(size, nz):
	"""Return B x nz noise vector"""
	return torch.randn(size, nz)  # B x nz

def make_y(size, ny, value=None):
	"""Return B condition vector"""
	if value is None:
		return torch.randint(ny, [size]).long()  # B (random value)
	else:
		return torch.LongTensor(size).fill_(value)  # B (given value)

def make_fixed_z(size, nz, ny):
	"""Return (B * ny) x nz noise vector (for visualization)"""
	z = make_z(size, nz)  # B x nz
	return torch.cat([z] * ny, dim=0)  # (B x ny) x nz

def make_fixed_y(size, ny):
	"""Return (B * ny) condition vector (for visualization)"""
	y = [torch.LongTensor(size).fill_(i) for i in range(ny)]  # list of B tensors
	return torch.cat(y, dim=0)  # (B * ny)


"""""""""""""""	
Helper functions (I/O)
"""""""""""""""

def count_classes(dataset, class_num):
	count = [0] * class_num
	for _, y in dataset:
		count[y] += 1
	return count

def save_to_logger(logger, info, step):
	for key, val in info.items():
		if isinstance(val, np.ndarray):
			logger.image_summary(key, val, step)
		else:
			logger.scalar_summary(key, val, step)

def normalize_info(info):
	num = info.pop('num')
	for key, val in info.items():
		info[key] /= num
	return info

def gold_score(netD, x, y, eps=1e-6):
	out_D, out_C = netD(x)  # B x 1, B x nc
	out_C = torch.softmax(out_C, dim=1)  # B x nc
	score_C = torch.log(out_C[torch.arange(len(out_C)), y] + eps)  # B
	return out_D.view(-1) + score_C  # B

def entropy(outs, eps=0):
	probs = F.softmax(outs, dim=1)  # B x nc
	entropy = -(probs * torch.log(probs + eps)).sum(-1)  # B
	return entropy  # B

def accuracy(out, tgt):
	_, pred = out.max(1)
	acc = pred.eq(tgt).sum().item() / len(out)
	return acc

def to_numpy_image(x):
	# convert torch tensor [-1,1] to numpy image [0,255]
	x = x.cpu().numpy().transpose(0, 2, 3, 1)  # C x H x W -> H x W x C
	x = ((x + 1) / 2).clip(0, 1)  # [-1,1] -> [0,1]
	x = (x * 255).astype(np.uint8)  # uint8 numpy image
	return x

