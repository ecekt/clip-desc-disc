import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, normaltest, mannwhitneyu
from collections import Counter

# ttest
ttest_ind = ttest_ind
# CLIP Listener GOLD
results_file = 'results/clip_listener_GOLD/full_pb_cliplistener_GOLD_chain_rank_scores.pt'
# results_file = 'results/clip_listener_domains/full_pb_cliplistener_domains_chain_rank_scores.pt'
# results_file = ''

print('Results file:', results_file + '\n')
chain_rank_scores = torch.load(results_file)

print([np.mean(chain_rank_scores[r]) for r in chain_rank_scores])

print('CLIP Listener GOLD Utterance rank 0 vs. 1 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[1]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')

print('CLIP Listener GOLD Utterance rank 0 vs. 2 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[2]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')

print('CLIP Listener GOLD Utterance rank 0 vs. 3 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')

print('CLIP Listener GOLD Utterance rank 1 vs. 2 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[1], chain_rank_scores[2]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')

print('CLIP Listener GOLD Utterance rank 1 vs. 3 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[1], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')

print('CLIP Listener GOLD Utterance rank 2 vs. 3 for resolution accuracy')
pvalue = ttest_ind(chain_rank_scores[2], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 3), '\n')
