import torch
import torch.utils.data as data_utils
from utils import entropy



def gold_acquistiion(pool, netD, args, device):
	def gold_score_unlabel(x):
		out_D, out_C = netD(x)  # B x 1, B x nc
		score_C = entropy(out_C)  # B
		return out_D.view(-1) + score_C  # B

	query_idx = score_based_acquisition(pool, gold_score_unlabel, args, device)
	return query_idx


def score_based_acquisition(pool, score_func, args, device):
	loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)

	scores = []
	for batch_idx, (real_x, _) in enumerate(loader):
		with torch.no_grad():
			real_x = real_x.to(device)  # B x nc x H x W
			score = score_func(real_x).cpu().numpy()  # B
			for i in range(len(score)):
				idx = batch_idx * args.pool_batch_size + i  # index in dataset
				scores.append((score[i], idx))

	query_idx = [x[1] for x in scores[-args.per_size:]]  # maximum values
	query_idx = [pool.indices[i] for i in query_idx]  # pool idx -> base_dataset idx

	return query_idx

