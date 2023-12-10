import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, normaltest, mannwhitneyu
from collections import Counter

# ttest
ttest_ind = ttest_ind

# CLIPSCORE COSINE
results_file = 'results/clipscore/full_pb_clipscore_chain_rank_scores.pt'

print('Results file:', results_file + '\n')
chain_rank_scores = torch.load(results_file)

print([round(np.median(chain_rank_scores[r]),3) for r in chain_rank_scores])
print([round(np.mean(chain_rank_scores[r]),3) for r in chain_rank_scores])

print('CLIPSCORE COSINE Utterance rank 0 vs. 1 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[1]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')

print('CLIPSCORE COSINE Utterance rank 0 vs. 2 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[2]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')

print('CLIPSCORE COSINE Utterance rank 0 vs. 3 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[0], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')

print('CLIPSCORE COSINE Utterance rank 1 vs. 2 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[1], chain_rank_scores[2]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')

print('CLIPSCORE COSINE Utterance rank 1 vs. 3 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[1], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')

print('CLIPSCORE COSINE Utterance rank 2 vs. 3 for descriptiveness')
pvalue = ttest_ind(chain_rank_scores[2], chain_rank_scores[3]).pvalue
print('ttest ind', 'pvalue', round(pvalue, 6), '\n')
