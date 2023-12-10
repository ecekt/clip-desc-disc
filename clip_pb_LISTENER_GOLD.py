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

if not os.path.exists('results/clip_listener_GOLD'):
    os.mkdir('results/clip_listener_GOLD')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

chain_rank_scores = defaultdict(list)
sentence_length_scores = defaultdict(list)
sentence_length_content_scores = defaultdict(list)

chain_lastrank_scores = defaultdict(list)

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

with open('full_pb_clip_listener_GOLD.txt', 'w') as f:

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
        visual_context_processed = []

        utt_rank = 0

        for utt in ch['utterances']:

            utt_str = full_utts[tuple(utt)]['utterance']
            # don't use the already tokenized version from RRR
            #utterances.append(utt_str)

            visual_context = full_utts[tuple(utt)]['image_set']
            visual_context_processed = torch.cat([clip_preprocessed_images[v_id] for v_id in visual_context])

            text = clip.tokenize(utt_str).to(device)

            with torch.no_grad():
                # image_features = model.encode_image(image)
                # text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(visual_context_processed, text)
                probs = logits_per_text.softmax(dim=-1).cpu().numpy()

            # print('target img', ch['target'])
            target_str = 'target img' + ' ' + ch['target'] + '\n'
            f.write(target_str)

            tokenized_utt = tweet_tokenizer.tokenize(utt_str)
            sent_len = len(tokenized_utt)
            sent_content_len = len([word for word in tokenized_utt if word not in stopwords.words('english')])
            # punctuation marks included
            # don't forget 'english'

            predicted_img = np.argmax(probs[0])
            predicted_str = 'predicted ' + visual_context[predicted_img]
            f.write(predicted_str)

            if predicted_img == visual_context.index(ch['target']):

                chain_rank_scores[utt_rank].append(1)
                sentence_length_scores[sent_len].append(1)
                sentence_length_content_scores[sent_content_len].append(1)

                if utt_rank == (len(ch['utterances']) - 1):
                    chain_lastrank_scores[utt_rank].append(1)

            else:

                chain_rank_scores[utt_rank].append(0)
                sentence_length_scores[sent_len].append(0)
                sentence_length_content_scores[sent_content_len].append(0)

                if utt_rank == (len(ch['utterances']) - 1):
                    chain_lastrank_scores[utt_rank].append(0)

            # print()
            f.write('\n')
            utt_rank += 1

print('chain rank')
for r in sorted(chain_rank_scores):
    print(r, np.mean(chain_rank_scores[r]), np.std(chain_rank_scores[r]))

print('sentence full length')
for l in sorted(sentence_length_scores):
    print(l, np.mean(sentence_length_scores[l]), np.std(sentence_length_scores[l]))

print('sentence content length')
for l in sorted(sentence_length_content_scores):
    print(l, np.mean(sentence_length_content_scores[l]), np.std(sentence_length_content_scores[l]))

print('chain last rank')
for r in sorted(chain_lastrank_scores):
    print(r, np.mean(chain_lastrank_scores[r]), np.std(chain_lastrank_scores[r]))

torch.save(chain_rank_scores, 'results/clip_listener_GOLD/full_pb_cliplistener_GOLD_chain_rank_scores.pt')
torch.save(sentence_length_scores, 'results/clip_listener_GOLD/full_pb_cliplistener_GOLD_full_length_scores.pt')
torch.save(sentence_length_content_scores, 'results/clip_listener_GOLD/full_pb_cliplistener_GOLD_content_length_scores.pt')
torch.save(chain_lastrank_scores, 'results/clip_listener_GOLD/full_pb_cliplistener_GOLD_chain_lastrank_scores.pt')
