#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" evaluation scripts for KBC and pathQuery tasks """
import json
import logging
import collections
import numpy as np

logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
	datefmt='%m/%d/%Y %H:%M:%S',
	level=logging.INFO)
logger = logging.getLogger(__name__)


def kbc_batch_evaluation(eval_i, all_examples, batch_results, tt):
	# dict
	r_hts_idx = collections.defaultdict(list)
	scores_head = collections.defaultdict(list)
	scores_tail = collections.defaultdict(list)
	batch_r_hts_cnt = 0
	print(np.array(batch_results).shape)
	b_size = len(batch_results)
	for j in range(b_size):
		# for each batch
		result = batch_results[j]
		# examples
		i = eval_i + j
		example = all_examples[i]
		#assert len(example.token_ids) == 3, "For kbc task each example consists of 3 tokens"
		
		#h, r, t = example.token_ids
		_mask_type = example.mask_type
		
		if _mask_type == "MASK_HEAD":
			h = example.token_ids[0]
			r = example.token_ids[1]
			t = example.token_ids[2]
			#print(example.token_ids)
			#print(h, r, t)
		elif _mask_type == "MASK_TAIL":
			h = example.token_ids[0]
			r = example.token_ids[-2]
			t = example.token_ids[-1]
		
		if i % 2 == 0:
			r_hts_idx[r].append((h, t))
			batch_r_hts_cnt += 1
		if _mask_type == "MASK_HEAD":
			scores_head[(r, t)] = result
		elif _mask_type == "MASK_TAIL":
			scores_tail[(r, h)] = result
		else:
			raise ValueError("Unknown mask type in prediction example:%d" % i)

	rank = {}
	f_rank = {}
	for r, hts in r_hts_idx.items():
		r_rank = {'head': [], 'tail': []}
		r_f_rank = {'head': [], 'tail': []}
		for h, t in hts:
			scores_t = scores_tail[(r, h)][:]
			# sort
			sortidx_t = np.argsort(scores_t)[::-1]
			r_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1)
			# filter
			rm_idx = tt[r]['ts'][h]
			rm_idx = [i for i in rm_idx if i != t] # need removed
			for i in rm_idx:
				scores_t[i] = -np.Inf # score = -inf
			sortidx_t = np.argsort(scores_t)[::-1] # reverse
			r_f_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1) # rank_id + 1

			scores_h = scores_head[(r, t)][:]
			sortidx_h = np.argsort(scores_h)[::-1]
			#print(sortidx_h)
			r_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)

			rm_idx = tt[r]['hs'][t]
			rm_idx = [i for i in rm_idx if i != h]
			for i in rm_idx:
				scores_h[i] = -np.Inf
			sortidx_h = np.argsort(scores_h)[::-1]
			r_f_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)
		rank[r] = r_rank
		f_rank[r] = r_f_rank

	h_pos = [p for k in rank.keys() for p in rank[k]['head']]
	t_pos = [p for k in rank.keys() for p in rank[k]['tail']]
	f_h_pos = [p for k in f_rank.keys() for p in f_rank[k]['head']]
	f_t_pos = [p for k in f_rank.keys() for p in f_rank[k]['tail']]

	ranks = np.asarray(h_pos + t_pos)
	f_ranks = np.asarray(f_h_pos + f_t_pos)
	return ranks, f_ranks


def compute_kbc_metrics(rank_li, frank_li, output_evaluation_result_file):
	""" combine the kbc rank results from batches into the final metrics """
	# flatten
	rank_rets = np.array(rank_li).ravel()
	frank_rets = np.array(frank_li).ravel()
	mrr = np.mean(1.0 / rank_rets)
	fmrr = np.mean(1.0 / frank_rets)

	hits1 = np.mean(rank_rets <= 1.0)
	hits3 = np.mean(rank_rets <= 3.0)
	hits10 = np.mean(rank_rets <= 10.0)
	# filtered metrics
	fhits1 = np.mean(frank_rets <= 1.0)
	fhits3 = np.mean(frank_rets <= 3.0)
	fhits10 = np.mean(frank_rets <= 10.0)

	eval_result = {
		'mrr': mrr,
		'hits1': hits1,
		'hits3': hits3,
		'hits10': hits10,
		'fmrr': fmrr,
		'fhits1': fhits1,
		'fhits3': fhits3,
		'fhits10': fhits10
	}
	with open(output_evaluation_result_file, "w") as fw:
		fw.write(json.dumps(eval_result, indent=4) + "\n")
	return eval_result


