import matplotlib.cm
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import sem


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

        sd = np.std(dictionary[k])
        se = sd / np.sqrt(len(dictionary[k]))
        print(sd, round(se, 4), round(sem(dictionary[k]), 4))
        #assert se == sem(dictionary[k])

        sds.append(se)

        # this plots everything

        # ys.extend(dictionary[k])
        # xs.extend([k] * len(dictionary[k]))

    return xs, ys, sds


font = {'size': 12}

matplotlib.rc('font', **font)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# CLIPScore alignment
chain_rank_scores = torch.load('results/clipscore_GOLD/full_pb_clipscore_GOLD_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x) - 0.2, y, 0.2, alpha=0.9,
        label='PB-GOLD',
        color=plt.cm.tab10.colors[0])
plt.errorbar(np.asarray(x) - 0.2, y, yerr=sd, fmt='.', color='black')

chain_rank_scores = torch.load('results/clipscore_COCOPB_GOLD/full_pb_clipscore_COCOPB_GOLD_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x), y, 0.2, alpha=0.8,
        label='COCO',
        color='mediumseagreen',
        hatch="/")
plt.errorbar(np.asarray(x), y, yerr=sd, fmt='.', color='black')

chain_rank_scores = torch.load('results/clipscore_TMM/full_pb_clipscore_TMM_chain_rank_scores.pt')
x, y, sd = create_x_y(chain_rank_scores)
print([round(d,3) for d in y])
plt.bar(np.asarray(x) + 0.2, y, 0.2, alpha=0.7,
        label='IDS',
        color='gold')
plt.errorbar(np.asarray(x) + 0.2, y, yerr=sd, fmt='.', color='black')

plt.title('Descriptiveness')
plt.xlabel('Utterance rank')
plt.ylabel('CLIPScore')
plt.xticks([1,2,3,4])
plt.ylim([0.53, 0.83])
plt.legend(loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig('alignment.pdf', dpi=300)
plt.close()

# image = Image.open('clipscore_all.png').convert("L")
# plt.imshow(image, cmap='gray')
# plt.show()

