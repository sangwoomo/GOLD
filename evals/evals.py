import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import network
from utils import make_z, make_y
from utils import accuracy, normalize_info


"""""""""""""""
Evaluation metrics
"""""""""""""""

def get_base_message(epoch, info):
	message = "Epoch: {}".format(epoch)
	message += "  G: {:.4f}".format(info['loss_G'])
	message += "  G (cls): {:.4f}".format(info['loss_G_cls'])
	message += "  D (real): {:.4f}".format(info['loss_D_real'])
	message += "  D (fake): {:.4f}".format(info['loss_D_fake'])
	message += "  C (real): {:.4f}".format(info['loss_C_real'])
	message += "  C (fake): {:.4f}".format(info['loss_C_fake'])
	return message


def adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_period=20, lr_decay_rate=0.1):
	lr = base_lr * (lr_decay_rate ** (epoch // lr_decay_period))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train_classifier(netG, args, device, testset=None):
	print('\nTraining a classifier')
	loader = data_utils.DataLoader(range(args.netC_per_epoch), batch_size=args.netC_batch_size, shuffle=True, num_workers=8)
	netC = network.LeNet(args.image_size, args.nc, args.ny).to(device)
	optimizerC = optim.Adam(netC.parameters(), lr=args.netC_lr, betas=(args.netC_beta1, args.netC_beta2))
	criterionCE = nn.CrossEntropyLoss().to(device)

	for epoch in range(1, args.netC_epochs + 1):
		adjust_learning_rate(optimizerC, epoch, args.netC_lr, args.netC_lr_period)
		info = {'num': 0, 'loss_C': 0, 'acc': 0}

		# train network
		netC.train()
		for i, x in enumerate(loader):
			# forward
			fake_z = make_z(len(x), args.nz).to(device)  # B x nz
			fake_y = make_y(len(x), args.ny).to(device)  # B
			with torch.no_grad():
				fake_x = netG(fake_z, fake_y)  # B x nc x H x W
			out_fake = netC(fake_x)  # B x nc
			loss_C = criterionCE(out_fake, fake_y)
			acc = accuracy(out_fake, fake_y)

			# backward
			optimizerC.zero_grad()
			loss_C.backward()
			optimizerC.step()

			# update loss info
			info['num'] += 1
			info['loss_C'] += loss_C.item()
			info['acc'] += acc

		# evaluate performance
		info = normalize_info(info)
		message = "Epoch: {}  C: {:.4f}  acc (train): {:.4f}".format(epoch, info['loss_C'], info['acc'])

		if testset and epoch % args.netC_eval_period == 0:
			test_acc = eval_classifier(netC, args, device, testset)
			message += "  acc (test): {:.4f}".format(test_acc)

		print(message)
	print('')

	return netC


def eval_classifier(netC, args, device, testset):
	loader = data_utils.DataLoader(testset, batch_size=args.netC_eval_batch_size, shuffle=False, num_workers=8)
	netC.eval()

	info = {'num': 0, 'acc': 0}  # loss info
	for i, (real_x, real_y) in enumerate(loader):
		real_x = real_x.to(device)  # B x nc x H x W
		real_y = real_y.to(device)  # B
		with torch.no_grad():
			out = netC(real_x)  # B x nc
			_, pred = out.max(1)
			correct = pred.eq(real_y).sum().item()

		info['num'] += 1
		info['acc'] += correct / len(real_x)

	acc = info['acc'] / info['num']
	return acc



