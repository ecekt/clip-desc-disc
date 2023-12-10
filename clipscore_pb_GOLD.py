import torch
import clip
from PIL import Image

import json
import pickle
from collections import defaultdict
import numpy as np

from nltk import TweetTokenizer
tweet_tokenizer = TweetTokenizer(preserve_case=False)

#import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(stopwords.words('english'))

import os

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('results/clipscore_GOLD'):
    os.mkdir('results/clipscore_GOLD')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean

# weight for CLIPScore (they use 2.5)
# TODO check range of scores with CLIPscore
weight = 2.5


def calculate_clipscore(text_features, img_features, weight):

    # we get multiple text features (for the utterances in the whole chain)
    clipscores = []
    for tx in text_features:

        cos_cross = cosine_similarity(tx.unsqueeze(0), img_features).item()

        if cos_cross < 0:
            print('neg', cos_cross)
        elif cos_cross > 1:
            print('gt one', cos_cross)

        clipscore = weight * max(cos_cross, 0) # since they take the max between cos and 0, no negs

        clipscores.append(clipscore)

    return clipscores


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

chain_rank_scores = defaultdict(list)
sentence_length_scores = defaultdict(list)
sentence_length_content_scores = defaultdict(list)

with open('data/gold_text_chains_CLIP.json', 'r') as f:
    full_chains = json.load(f)

with open('data/gold_text_utterances_CLIP.pickle', 'rb') as f:
    full_utts = pickle.load(f)

with open('coco_id2file.json', 'r') as f:
    coco_id2file = json.load(f)

# don't preprocess images again and again
with open('data/clip_preprocessed_images.pickle', 'rb') as f:
    clip_preprocessed_images = pickle.load(f)

count_ch = 0

print('len chains', len(full_chains))  # 18060 full

with open('full_pb_clipscore_GOLD.txt', 'w') as f:

    for ch in full_chains:

        f.write('chain ' + str(count_ch) + '\n')

        count_ch += 1
        if count_ch % 100 == 0:
            print(count_ch)

        image = clip_preprocessed_images[ch['target']]
        # image_path = 'photobook_coco_images/images/' + coco_id2file[ch['target']]
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        #
        # assert torch.all(torch.eq(clip_preprocessed_images[ch['target']], image))

        utterances = []

        for utt in ch['utterances']:

            utt_str = full_utts[tuple(utt)]['utterance']
            # don't use the already tokenized version from RRR
            utterances.append(utt_str)

        text = clip.tokenize(utterances).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # THESE TWO look the same, but they are not exactly the same
            # due to minor differences in precision e-08
            # model.encode_text(text[0].unsqueeze(0)) != text_features[0].unsqueeze(0)

            probs = calculate_clipscore(text_features, image_features, weight)

        # print('target img', ch['target'])
        target_str = 'target img' + ' ' + ch['target'] + '\n'
        f.write(target_str)

        for i in range(len(utterances)):

            # print(probs[0][i], utterances[i])
            prob_str = str(probs[i]) + ' ' + utterances[i] + '\n'
            f.write(prob_str)

            chain_rank_scores[i].append(probs[i])

            tokenized_utt = tweet_tokenizer.tokenize(utterances[i])
            sent_len = len(tokenized_utt)
            sent_content_len = len([word for word in tokenized_utt if word not in stopwords.words('english')])
            # punctuation marks included
            # don't forget 'english'

            sentence_length_scores[sent_len].append(probs[i])
            sentence_length_content_scores[sent_content_len].append(probs[i])

        # print()
        f.write('\n')

print('chain rank')
for r in sorted(chain_rank_scores):
    print(r, np.mean(chain_rank_scores[r]), np.std(chain_rank_scores[r]))

print('sentence full length')
for l in sorted(sentence_length_scores):
    print(l, np.mean(sentence_length_scores[l]), np.std(sentence_length_scores[l]))

print('sentence content length')
for l in sorted(sentence_length_content_scores):
    print(l, np.mean(sentence_length_content_scores[l]), np.std(sentence_length_content_scores[l]))

torch.save(chain_rank_scores, 'results/clipscore_GOLD/full_pb_clipscore_GOLD_chain_rank_scores.pt')
torch.save(sentence_length_scores, 'results/clipscore_GOLD/full_pb_clipscore_GOLD_full_length_scores.pt')
torch.save(sentence_length_content_scores, 'results/clipscore_GOLD/full_pb_clipscore_GOLD_content_length_scores.pt')
