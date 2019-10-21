import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import network
from utils import make_z, make_y
from utils import normalize_info, accuracy



def get_score_stats(netG, netD, sample=50000):
	score_D = sample_scores(netG, netD, wd=1, wc=0, sample_size=sample)
	score_C = sample_scores(netG, netD, wd=0, wc=1, sample_size=sample)

	M = np.exp(np.max(score_D))
	w = np.std(score_D) / np.sqrt(np.mean(np.square(score_C)))

	return M, w


def sample_scores(netG, netD, nz=100, ny=10, wd=1, wc=1, sample_size=50000, batch_size=100):
	scores = []
	for i in range(sample_size // batch_size):
		z = make_z(batch_size, nz).cuda()
		y = make_y(batch_size, ny).cuda()
		with torch.no_grad():
			x = netG(z, y)
			s = gold(netD, x, y, wd, wc)
		scores.append(s)
	scores = np.concatenate(scores, axis=0)
	return scores


def gold(netD, x, y, wd=1, wc=1, verbose=False):
	with torch.no_grad():
		out_D, out_C = netD(x)  # B x 1, B x nc

	score_D = out_D.view(-1) * wd
	out_C = torch.softmax(out_C, dim=1)
	out_C = out_C[torch.arange(len(out_C)), y]
	score_C = torch.log(out_C) * wc

	if verbose:
		plt.hist(score_D.cpu().numpy())
		plt.hist(score_C.cpu().numpy())

	return (score_D + score_C).cpu().numpy()


def drs(netG, netD, num_samples=10, perc=10, nz=100, ny=10, batch_size=100, eps=1e-6):
	M, w = get_score_stats(netG, netD)
	ones = np.ones(batch_size).astype('int64')

	images = [[] for _ in range(ny)]
	for cls in range(10):
		while len(images[cls]) < num_samples:
			z = make_z(batch_size, nz).cuda()
			y = make_y(batch_size, ny, cls).cuda()
			with torch.no_grad():
				x = netG(z, y)
				r = np.exp(gold(netD, x, y, 1, w))

			p = np.minimum(ones, r/M)
			f = np.log(p + eps) - np.log(1 - p + eps)  # inverse sigmoid
			f = (f - np.percentile(f, perc))
			p = [1 / (1 + math.exp(-x)) for x in f]  # sigmoid
			accept = np.random.binomial(ones, p)

			for i in range(batch_size):
				if accept[i] and len(images[cls]) < num_samples:
					images[cls].append(x[i].detach().cpu())

	images = torch.stack([x for l in images for x in l])
	return images


def adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_period=20, lr_decay_rate=0.1):
	lr = base_lr * (lr_decay_rate ** (epoch // lr_decay_period))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def fitting_capacity(samples, testset, nc=1, ny=10, epochs=40, eval_period=10, verbose=False):
	netC = network.LeNet(32, nc, ny)
	netC = nn.DataParallel(netC, [0]).cuda()
	optimizerC = optim.Adam(netC.parameters(), lr=0.001, betas=(0.5, 0.999))
	criterionCE = nn.CrossEntropyLoss()

	loader = data_utils.DataLoader(samples, batch_size=128, shuffle=True, num_workers=8)
	test_acc = 0
	for epoch in range(1, epochs + 1):
		adjust_learning_rate(optimizerC, epoch, 0.001, epochs//2)
		info = {'num': 0, 'loss_C': 0, 'acc': 0}

		# train network
		netC.train()
		for i, (x, y) in enumerate(loader):
			# forward
			x = x.cuda()
			y = y.cuda()
			out = netC(x)  # B x nc
			loss_C = criterionCE(out, y)

			# backward
			optimizerC.zero_grad()
			loss_C.backward()
			optimizerC.step()

			# update loss info
			info['num'] += 1
			info['loss_C'] += loss_C.item()
			info['acc'] += accuracy(out, y)

		# evaluate performance
		info = normalize_info(info)
		message = "Epoch: {}  C: {:.4f}  acc (train): {:.4f}".format(epoch, info['loss_C'], info['acc'])
		if epoch % eval_period == 0:
			test_acc = eval_classifier(netC, testset)
			message += "  acc (test): {:.4f}".format(test_acc)

		if verbose:
			print(message)

	return test_acc


def eval_classifier(netC, testset):
	loader = data_utils.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
	netC.eval()

	info = {'num': 0, 'acc': 0}  # loss info
	for i, (x, y) in enumerate(loader):
		x = x.cuda()  # B x nc x H x W
		y = y.cuda()  # B
		with torch.no_grad():
			pred = netC(x).max(1)[1]
			correct = pred.eq(y).sum().item()

		info['num'] += 1
		info['acc'] += correct / len(x)

	acc = info['acc'] / info['num']
	return acc

