import numpy as np
import sklearn.datasets
import torch
import torch.utils.data as data_utils
from torchvision import datasets
from torchvision import transforms as T


def get_transform(image_size, transform_type):
	if transform_type == 'none':
		return lambda x: x
	elif transform_type == 'base':
		return T.Compose([
			T.Resize(image_size),
			T.CenterCrop(image_size),
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	elif transform_type == 'random_crop':
		base_size = int(image_size * 1.1)
		return T.Compose([
			T.Resize(base_size),
			T.RandomCrop(image_size),
			T.Resize(image_size),
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	elif transform_type == 'random_crop_and_flip':
		base_size = int(image_size * 1.1)
		return T.Compose([
			T.Resize(base_size),
			T.RandomCrop(image_size),
			T.RandomHorizontalFlip(),
			T.Resize(image_size),
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	else:
		raise NotImplementedError


def load_base_dataset(args):
	if args.dataset == 'synthetic':
		base_dataset, test_dataset = generate_synthetic_dataset(args)
	elif args.dataset == 'mnist':
		base_dataset = datasets.MNIST('./dataset/mnist', train=True, download=True)
		test_dataset = datasets.MNIST('./dataset/mnist', train=False, download=True)
	elif args.dataset == 'fmnist':
		base_dataset = datasets.FashionMNIST('./dataset/fmnist', train=True, download=True)
		test_dataset = datasets.FashionMNIST('./dataset/fmnist', train=False, download=True)
	elif args.dataset == 'svhn':
		base_dataset = datasets.SVHN('./dataset/svhn', split='train', download=True)
		test_dataset = datasets.SVHN('./dataset/svhn', split='test', download=True)
	elif args.dataset == 'cifar10':
		base_dataset = datasets.CIFAR10('./dataset/cifar10', train=True, download=True)
		test_dataset = datasets.CIFAR10('./dataset/cifar10', train=False, download=True)
	elif args.dataset == 'stl10':
		base_dataset = datasets.STL10('./dataset/stl10', split='train', download=True)
		test_dataset = datasets.STL10('./dataset/stl10', split='test', download=True)
	elif args.dataset == 'lsun':
		train_transform = get_transform(args.image_size, args.train_transform)
		test_transform = get_transform(args.image_size, args.test_transform)
		base_dataset = datasets.LSUN('./dataset/lsun', classes='val', transform=train_transform)
		test_dataset = datasets.LSUN('./dataset/lsun', classes='val', transform=test_transform)
	else:
		raise NotImplementedError

	return base_dataset, test_dataset


def generate_synthetic_dataset(args):
	n_base = args.n_samples_base
	n_test = args.n_samples_test

	full_dataset = sklearn.datasets.make_blobs(n_base + n_test, cluster_std=0.5, centers=6)
	full_dataset = (full_dataset[0], full_dataset[1] % 2)

	xs = torch.FloatTensor(full_dataset[0][:n_base])
	ys = torch.LongTensor(full_dataset[1][:n_base])
	base_dataset = data_utils.TensorDataset(xs, ys)

	xs = torch.FloatTensor(full_dataset[0][n_base:])
	ys = torch.LongTensor(full_dataset[1][n_base:])
	test_dataset = data_utils.TensorDataset(xs, ys)

	return base_dataset, test_dataset


def split_dataset(base_dataset, num_classes, init_size, val_size):
	shuffled_idx = np.random.permutation(len(base_dataset))
	train_idx = pick_samples(base_dataset, num_classes, shuffled_idx, init_size)

	shuffled_idx = list(set(shuffled_idx) - set(train_idx))
	val_idx = pick_samples(base_dataset, num_classes, shuffled_idx, val_size)

	pool_idx = list(set(shuffled_idx) - set(val_idx))

	return train_idx, val_idx, pool_idx


def pick_samples(base_dataset, num_classes, base_idx, size):
	sub_idx = []
	for cls in range(num_classes):
		for idx in base_idx:
			if len(sub_idx) == (size // num_classes) * (cls + 1):
				break
			if base_dataset[idx][1] == cls:
				sub_idx.append(idx)
	return sub_idx


