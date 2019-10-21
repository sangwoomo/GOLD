import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import data
import network
import train
import evals
import query
from utils import *



def add_parser(parser):
	# base arguments
	parser.add_argument('--name', type=str, default='temp', help='experiment name')
	parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--dataset', type=str, default='mnist', help='dataset')
	parser.add_argument('--image_size', type=int, default=32, help='image size (default: 32)')
	parser.add_argument('--train_transform', type=str, default='random_crop', help='data augmentaion (training)')
	parser.add_argument('--test_transform', type=str, default='base', help='data augmentation (test)')
	parser.add_argument('--n_samples_base', type=int, default=1000, help='number of base dataset (default: 1000)')
	parser.add_argument('--n_samples_test', type=int, default=1000, help='number of test dataset (default: 1000)')
	parser.add_argument('--mode', type=str, default='acgan', help='train method (acgan|acgan_gold|etc.)')
	parser.add_argument('--network', type=str, default='acgan_sn', help='network architecture for GAN')
	parser.add_argument('--nz', type=int, default=100, help='dimension of noise vector (default: 100)')
	parser.add_argument('--ny', type=int, default=10, help='number of classes (default: 10)')
	parser.add_argument('--nc', type=int, default=1, help='number of channels of image (default: 1)')

	# training arguments
	parser.add_argument('--epochs', type=int, default=100, help='epochs (default: 100)')
	parser.add_argument('--per_epoch', type=int, default=10000, help='# of total training samples (default: 10000)')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
	parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam (default: 0.5)')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam (default: 0.999)')
	parser.add_argument('--lambda_C_fake', type=float, default=0.1, help='weight for loss_C_fake (default: 0.1)')
	parser.add_argument('--use_final_model', action='store_true', help='use final instead of best (default: False)')
	parser.add_argument('--compare_metric', type=str, default='cap_val', help='compare models (default: cap_val)')

	# evaluation arguments
	parser.add_argument('--eval_period', type=int, default=10, help='evaluation period for heavy stuffs (default: 10)')
	parser.add_argument('--netC_network', type=str, default='lenet', help='network architecture for classifier')
	parser.add_argument('--netC_epochs', type=int, default=40, help='epochs for classifier (default: 40)')
	parser.add_argument('--netC_per_epoch', type=int, default=10000, help='per epoch for classifier (default: 10000)')
	parser.add_argument('--netC_batch_size', type=int, default=128, help='batch size for classifier (default: 128)')
	parser.add_argument('--netC_lr', type=float, default=0.001, help='learning rate for classifier (default: 0.001)')
	parser.add_argument('--netC_lr_period', type=int, default=20, help='lr decay period for classifier (default: 20)')
	parser.add_argument('--netC_beta1', type=float, default=0.5, help='beta1 for classifier (default: 0.5)')
	parser.add_argument('--netC_beta2', type=float, default=0.999, help='beta2 for classifier (default: 0.999)')
	parser.add_argument('--netC_eval_period', type=int, default=10, help='eval period ofr classifier (default: 10)')
	parser.add_argument('--netC_eval_batch_size', type=int, default=1000, help='eval batch size for classifier (default: 1000)')

	# query arguments
	parser.add_argument('--init_size', type=int, default=None, help='size of initial training set (default: None)')
	parser.add_argument('--per_size', type=int, default=None, help='size of query for each acquisition (default: None)')
	parser.add_argument('--max_size', type=int, default=None, help='size of maximum training set (default: None)')
	parser.add_argument('--val_size', type=int, default=100, help='size of validation set (default: 100)')
	parser.add_argument('--query_type', type=str, default='random', help='acquisition algorithm (random|maxent|etc.)')
	parser.add_argument('--pool_batch_size', type=int, default=1000, help='batch size for query selection (default: 1000)')
	parser.add_argument('--reinit_type', type=str, default='cont_G', help='re-initialization for each query iteration')

	return parser


class BaseModel(object):
	def __init__(self, args):
		self.args = args
		self.set_device()
		self.logger = Logger('./logs/{}'.format(self.args.name))

	def set_device(self):
		str_ids = self.args.gpu_ids.split(',')
		self.gpu_ids = []
		for str_id in str_ids:
			if int(str_id) >= 0:
				self.gpu_ids.append(int(str_id))
		if len(self.gpu_ids) > 0:
			torch.cuda.set_device(self.gpu_ids[0])
			self.device = torch.device('cuda:{}'.format(self.args.gpu_ids[0]))
		else:
			self.device = torch.device('cpu')

	"""""""""""""""
	Run model
	"""""""""""""""

	def run(self):
		# initialize setting
		self.init_data()
		self.init_model()

		# run experiment
		self.it = 0  # iteration (start from 0)
		while len(self.trainset) <= self.args.max_size:
			self.it += 1  # update iteration
			self.show_status()

			# train network
			if len(self.trainset) > 0:
				self.reinit_model()
				self.train()

			# add new query
			if len(self.trainset) < self.args.max_size:
				self.add_query()
			else:
				break

	"""""""""""""""
	Initialize model
	"""""""""""""""

	def init_data(self):
		print('Initialize dataset...')
		self.train_transform = data.get_transform(self.args.image_size, self.args.train_transform)
		self.test_transform = data.get_transform(self.args.image_size, self.args.test_transform)

		# load base dataset
		self.base_dataset, self.test_dataset = data.load_base_dataset(self.args)
		self.base_dataset.transform = self.train_transform
		self.test_dataset.transform = self.test_transform

		# split to train/val/pool set
		if self.args.init_size is None:
			self.train_idx = list(range(len(self.base_dataset)))
			self.val_idx = []
			self.pool_idx = []
			self.args.init_size = len(self.base_dataset)
			self.args.per_size = 0
			self.args.max_size = len(self.base_dataset)
		else:
			self.train_idx, self.val_idx, self.pool_idx = data.split_dataset(
				self.base_dataset, self.args.ny, self.args.init_size, self.args.val_size)

		if self.args.max_size is None:
			self.args.per_size = 0
			self.args.max_size = self.args.init_size

		# define trainset and pool
		self.trainset = data_utils.Subset(self.base_dataset, self.train_idx)
		self.valset = data_utils.Subset(self.base_dataset, self.val_idx)
		self.pool = data_utils.Subset(self.base_dataset, self.pool_idx)

	def show_status(self):
		# update count
		if self.it == 1:
			self.count = count_classes(self.trainset, self.args.ny)
		self.prev_count = self.count
		self.count = count_classes(self.trainset, self.args.ny)

		message = '\n# of classes:'
		for i in range(self.args.ny):
			message += '  {}: {} (+{})'.format(i, self.count[i], self.count[i] - self.prev_count[i])
		message += '  sum: {}'.format(sum(self.count))
		print(message)

	def init_model(self):
		print('Initialize networks...')
		self._init_model_G()
		self._init_model_D()
		self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
		self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

	def reinit_model(self):
		if self.args.reinit_type == 'random':
			self._init_model_G()
			self._init_model_D()
			self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
			self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
		elif self.args.reinit_type == 'cont_G':
			self._init_model_D()
			self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
			self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
		else:
			raise NotImplementedError

	def _init_model_G(self):
		if self.args.dataset in ['synthetic']:
			self.netG = network.ACGAN_Toy_Generator(self.args.nz, self.args.nc, self.args.ny).to(self.device)
		elif self.args.dataset in ['mnist', 'fmnist']:
			self.netG = network.ACGAN_MNIST_Generator(self.args.nz, self.args.nc, self.args.ny).to(self.device)
		else:
			self.netG = network.ACGAN_CIFAR10_Generator(self.args.nz, self.args.nc, self.args.ny).to(self.device)

	def _init_model_D(self):
		if self.args.dataset in ['synthetic']:
			self.netD = network.ACGAN_Toy_Discriminator(self.args.nc, self.args.ny, use_sn=True).to(self.device)
		elif self.args.dataset in ['mnist', 'fmnist']:
			self.netD = network.ACGAN_MNIST_Discriminator(self.args.nc, self.args.ny, use_sn=True).to(self.device)
		else:
			self.netD = network.ACGAN_CIFAR10_Discriminator(self.args.nc, self.args.ny, use_sn=True).to(self.device)

	"""""""""""""""	
	Main functions
	"""""""""""""""

	def train(self):
		print('Train networks...')

		# train GAN networks
		best_epoch = 0
		best_score = 0
		metric = self.args.compare_metric
		for epoch in range(1, self.args.epochs + 1):
			info = self._train_sub(epoch)
			info = self._eval_sub(epoch, info)

			# save the best model
			if metric in info and info[metric] >= best_score:
				netG_best = self.netG.state_dict()
				netD_best = self.netD.state_dict()
				best_epoch = epoch
				best_score = info[metric]

		# load the best model
		if not self.args.use_final_model and best_epoch > 0:
			print('\nUse the best networks (epoch: {}, score: {:.3f})'.format(best_epoch, best_score))
			self.netG.load_state_dict(netG_best)
			self.netD.load_state_dict(netD_best)

		# save network
		netG_path = './logs/{}/netG_{}.pth'.format(self.args.name, self.it)
		netD_path = './logs/{}/netD_{}.pth'.format(self.args.name, self.it)
		torch.save(self.netG.cpu().state_dict(), netG_path)
		torch.save(self.netD.cpu().state_dict(), netD_path)
		self.netG = self.netG.to(self.device)
		self.netD = self.netD.to(self.device)

	def _train_sub(self, epoch):
		self.start_time_train = time.time()
		self.base_dataset.transform = self.train_transform
		self.netG.train()
		self.netD.train()

		model = {
			'net_G': self.netG,
			'net_D': self.netD,
			'optim_G': self.optimizerG,
			'optim_D': self.optimizerD,
		}

		# train networks by 1 epoch
		if self.args.mode == 'acgan_semi':
			return train.train_acgan_semi(self.trainset, self.pool, model, self.args, self.device)
		elif self.args.mode == 'acgan_semi_gold':
			if epoch <= self.args.epochs // 2:
				return train.train_acgan_semi(self.trainset, self.pool, model, self.args, self.device)
			else:
				return train.train_acgan_semi(self.trainset, self.pool, model, self.args, self.device, use_gold=True)
		else:
			raise NotImplementedError

	def _eval_sub(self, epoch, info):
		self.start_time_eval = time.time()
		self.base_dataset.transform = self.test_transform
		self.netG.eval()
		self.netD.eval()

		# compute evaluation metrics
		message = evals.get_base_message(epoch, info)
		if epoch % self.args.eval_period == 0:
			self.netC = evals.train_classifier(self.netG, self.args, self.device, testset=self.test_dataset)

			cap_test = evals.eval_classifier(self.netC, self.args, self.device, testset=self.test_dataset)
			message += "  cap (test): {:.4f}".format(cap_test)
			info['cap_test'] = cap_test

			if len(self.valset) > 0:
				cap_test = evals.eval_classifier(self.netC, self.args, self.device, testset=self.valset)
				message += "  cap (val): {:.4f}".format(cap_test)
				info['cap_val'] = cap_test

		eval_time = int(time.time() - self.start_time_eval)
		train_time = int(time.time() - self.start_time_train) - eval_time
		message += '  {:d}s/{:d}s elapsed'.format(train_time, eval_time)

		print(message)
		step = self.args.epochs * (self.it - 1) + epoch
		save_to_logger(self.logger, info, step)

		return info

	def add_query(self):
		print('\nSelect queries... (query type: {})'.format(self.args.query_type))
		start_time = time.time()
		self.netG.eval()
		self.netD.eval()

		# get query & update train/pool
		if self.args.query_type == 'random':
			query_idx = np.random.permutation(len(self.pool))[:self.args.per_size]
		elif self.args.query_type == 'gold':
			query_idx = query.gold_acquistiion(self.pool, self.netD, self.args, self.device)
		else:
			raise NotImplementedError

		self.train_idx = list(set(self.train_idx) | set(query_idx))
		self.pool_idx = list(set(self.pool_idx) - set(query_idx))

		self.trainset = data_utils.Subset(self.base_dataset, self.train_idx)
		self.pool = data_utils.Subset(self.base_dataset, self.pool_idx)

		# print computation time
		query_time = int(time.time() - start_time)
		print('{:d}s elapsed'.format(query_time))


def main():
	# get arguments
	parser = argparse.ArgumentParser()
	parser = add_parser(parser)
	args = parser.parse_args()

	# set random seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# define model and run
	model = BaseModel(args)
	model.run()


if __name__ == '__main__':
	main()


