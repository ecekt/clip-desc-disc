import torch
import clip
from PIL import Image

import json
import pickle
from collections import defaultdict, Counter
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from nltk import TweetTokenizer
tweet_tokenizer = TweetTokenizer(preserve_case=False)

#import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(stopwords.words('english'))

import os

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('results/clip_captionword_GOLD_RETEXISTS'):
    os.mkdir('results/clip_captionword_GOLD_RETEXISTS')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

chain_rank_scores = defaultdict(list)

with open('data/gold_text_chains_CLIP.json', 'r') as f:
    full_chains = json.load(f)

with open('data/gold_text_utterances_CLIP.pickle', 'rb') as f:
    full_utts = pickle.load(f)

vocab = []
text_tokens = defaultdict()
count_ch = 0

for ch in full_chains:

    count_ch += 1
    if count_ch % 100 == 0:
        print(count_ch)

    utterances = []

    for utt in ch['utterances']:

        utt_str = full_utts[tuple(utt)]['utterance']
        # don't use the already tokenized version from RRR
        utterances.append(utt_str)
        tokenized_utt = tweet_tokenizer.tokenize(utt_str)

        vocab.extend(tokenized_utt)

vocab = list(Counter(vocab).keys())
print(len(vocab))

with torch.no_grad():

    for v in vocab:

        if v not in text_tokens:
            v_str = clip.tokenize(v).to(device)
            v_features = model.encode_text(v_str)
            text_tokens[v] = v_features

print(len(text_tokens))
text = torch.cat([text_tokens[v] for v in text_tokens])
words = [v for v in text_tokens]
i2w = {i:words[i] for i in range(len(words))}

with open('coco_id2file.json', 'r') as f:
    coco_id2file = json.load(f)

# don't preprocess images again and again
with open('data/clip_preprocessed_images.pickle', 'rb') as f:
    clip_preprocessed_images = pickle.load(f)

count_ch = 0

print('len chains', len(full_chains))  # 18060 full


def closest_word(text_encs, vis_enc):

    scores = cosine_similarity(text_encs, vis_enc)
    index = torch.argmax(torch.tensor(scores))

    return index

with open('full_pb_clip_captionword_GOLD_RETEXISTS.txt', 'w') as f:

    for ch in full_chains:

        f.write('chain ' + str(count_ch) + '\n')
        f.write('game id ' + str(ch['game_id']) + '\n')

        count_ch += 1
        if count_ch % 100 == 0:
            print(count_ch)

        image = clip_preprocessed_images[ch['target']]
        # image_path = 'photobook_coco_images/images/' + coco_id2file[ch['target']]
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        #
        # assert torch.all(torch.eq(clip_preprocessed_images[ch['target']], image))

        utterances = []
        visual_context_processed = []

        utt_rank = 0

        for utt in ch['utterances']:

            utt_str = full_utts[tuple(utt)]['utterance']
            # don't use the already tokenized version from RRR
            #utterances.append(utt_str)

            visual_context = full_utts[tuple(utt)]['image_set']
            visual_target_processed = torch.cat([clip_preprocessed_images[v_id] for v_id in visual_context if v_id == ch['target']])
            visual_distractors_processed = torch.cat([clip_preprocessed_images[v_id] for v_id in visual_context if v_id != ch['target']])

            with torch.no_grad():
                target_image_features = model.encode_image(visual_target_processed)
                distractor_image_features = model.encode_image(visual_distractors_processed)

                mean_distractor_features = torch.mean(distractor_image_features, dim=0)

                difference_features = target_image_features - mean_distractor_features

                index_closest = closest_word(text, difference_features).item()

            # print('target img', ch['target'])
            target_str = 'target img' + ' ' + ch['target'] + '\n'
            f.write(target_str)

            f.write(utt_str + '\n')

            predicted_word = i2w[index_closest]
            predicted_str = 'predicted: ' + predicted_word
            f.write(predicted_str + '\n')

            if predicted_word.lower() in utt_str.lower():

                chain_rank_scores[utt_rank].append(1)
                f.write('is mentioned, rank ' + str(utt_rank) +
                        ' round ' + str(full_utts[tuple(utt)]['round_nr']) + '\n')

            else:

                chain_rank_scores[utt_rank].append(0)
                f.write('not mentioned, rank ' + str(utt_rank) +
                        ' round ' + str(full_utts[tuple(utt)]['round_nr']) + '\n')

            utt_rank += 1
            f.write('\n')

print('chain rank')
for r in sorted(chain_rank_scores):
    print(r, np.mean(chain_rank_scores[r]), np.std(chain_rank_scores[r]))

torch.save(chain_rank_scores, 'results/clip_captionword_GOLD_RETEXISTS/full_pb_clip_captionword_GOLD_RETEXISTS_chain_rank_scores.pt')
