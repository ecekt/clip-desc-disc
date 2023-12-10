import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib


def create_x_y(dictionary):

    xs = []
    ys = []
    sds = []

    for k in sorted(dictionary):

        if k > 3:
            break  # TMM has more

        # this plots mean
        xs.append(k + 1)
        ys.append(np.mean(dictionary[k]))
        sds.append(np.std(dictionary[k]))

        # this plots everything

        # ys.extend(dictionary[k])
        # xs.extend([k] * len(dictionary[k]))

    return xs, ys, sds

font = {'size': 12}

matplotlib.rc('font', **font)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# CLIPScore alignment
chain_rank_scores = torch.load('results/clip_listener_GOLD/full_pb_cliplistener_GOLD_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x) - 0.2, y, 0.2,  alpha=0.9, label='PB-GOLD')

chain_rank_scores = torch.load('results/clip_captionword_GOLD_EVAL/full_pb_clip_captionword_GOLD_EVAL_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x), y, 0.2, alpha=0.7, label='TARGET-CONTEXT')

chain_rank_scores = torch.load('results/clip_captionword_GOLD_ISOLATED_EVAL/full_pb_clip_captionword_GOLD_ISOLATED_EVAL_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x) + 0.200001, y, 0.2, alpha=0.7, label='TARGET ONLY')

plt.title('Discriminativeness')
plt.xlabel('Utterance rank')
plt.ylabel('Accuracy')
plt.xticks([1,2,3,4])
plt.ylim([0.45, 1.0])
plt.legend(loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig('resolution.pdf', dpi=300)
plt.close()
