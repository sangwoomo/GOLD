import torch
import torch.nn as nn
import torch.utils.data as data_utils
from copy import deepcopy

import network
from utils import make_z, make_y
from utils import gold_score, normalize_info



def train_acgan_semi(trainset, pool, model, args, device=None, use_gold=False):
	# preprocess dataset (labels of pool = -1)
	if args.dataset != 'lsun':
		pool = deepcopy(pool)

	if args.dataset == 'synthetic':
		ones = torch.ones(len(pool.dataset)).long()
		pool.dataset.tensors = (pool.dataset.tensors[0], -ones)
	else:
		ones = torch.ones(len(pool.dataset)).long()
		pool.dataset.train_labels = -ones

	if args.dataset == 'cifar10':
		trainset = deepcopy(trainset)
		trainset.dataset.train_labels = torch.tensor(trainset.dataset.train_labels).long()

	dataset = data_utils.ConcatDataset([trainset, pool])
	if len(dataset) < args.per_epoch:
		n_iter = args.per_epoch // len(dataset)
		dataset = data_utils.ConcatDataset([dataset] * n_iter)
	loader = data_utils.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

	# preprocess model
	netG = model['net_G']
	netD = model['net_D']
	optimizerG = model['optim_G']
	optimizerD = model['optim_D']

	# initialize criterion
	criterionGAN = network.GANLoss(reduction='none').to(device)
	criterionCE = nn.CrossEntropyLoss(reduction='none').to(device)

	# initialize loss info
	info = {'num': 0, 'loss_G': 0, 'loss_G_cls': 0, 'loss_D_real': 0, 'loss_D_fake': 0, 'loss_C_real': 0, 'loss_C_fake': 0}

	# train one epoch
	for i, (real_x, real_y) in enumerate(loader):
		idx_l = [i for i in range(len(real_x)) if real_y[i] != -1]

		# forward
		real_x = real_x.to(device)  # B x nc x H x W
		real_y = real_y.to(device)  # B
		fake_z = make_z(len(real_x), args.nz).to(device)  # B x nz
		fake_y = make_y(len(real_x), args.ny).to(device)  # B

		#########################
		# (1) Update D network
		#########################

		optimizerD.zero_grad()

		# real loss
		out_D, out_C = netD(real_x)  # B x 1, B x nc
		loss_D_real = torch.mean(criterionGAN(out_D, True))
		if len(idx_l) > 0:
			loss_C_real = torch.mean(criterionCE(out_C[idx_l], real_y[idx_l]))
		else:
			loss_C_real = 0

		# fake loss
		fake_x = netG(fake_z, fake_y)  # B x nc x H x W
		out_D, out_C = netD(fake_x.detach())  # B x 1, B x nc
		with torch.no_grad():
			gold = gold_score(netD, fake_x, fake_y)

		if use_gold:
			weight = gold
		else:
			weight = torch.ones(len(gold)).to(device)

		loss_D_fake = torch.mean(criterionGAN(out_D, False) * weight)
		if len(idx_l) > 0:
			loss_C_fake = torch.mean(criterionCE(out_C[idx_l], fake_y[idx_l]) * weight[idx_l]) * args.lambda_C_fake
		else:
			loss_C_fake = 0

		loss_D = loss_D_real + loss_D_fake + loss_C_real + loss_C_fake
		loss_D.backward()
		optimizerD.step()

		#########################
		# (2) Update G network
		#########################

		optimizerG.zero_grad()

		# GAN & classification loss
		fake_x = netG(fake_z, fake_y)  # B x nc x H x W
		out_D, out_C = netD(fake_x)  # B x 1, B x nc
		loss_G = torch.mean(criterionGAN(out_D, True))
		loss_G_cls = torch.mean(criterionCE(out_C, fake_y))

		# backward loss
		loss_G_total = loss_G + loss_G_cls
		loss_G_total.backward()
		optimizerG.step()

		# update loss info
		info['num'] += 1

		info['loss_G'] += loss_G.item()
		info['loss_G_cls'] += loss_G_cls.item()

		info['loss_D_real'] += loss_D_real.item()
		info['loss_D_fake'] += loss_D_fake.item()

		if len(idx_l) > 0:
			info['loss_C_real'] += loss_C_real.item()
			info['loss_C_fake'] += loss_C_fake.item()

	info = normalize_info(info)
	return info

