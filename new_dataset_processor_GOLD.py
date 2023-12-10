import json
import pickle

from nltk import TweetTokenizer
from collections import Counter, defaultdict
import csv

tweet_tokenizer = TweetTokenizer(preserve_case=False)
min_freq = 2


def process_data(data, split, min_freq=2):

    chain_dataset = []
    utterance_dataset = defaultdict()

    chains_path = 'data/' + split + '_text_chains_CLIP.json'
    utterances_path = 'data/' + split + '_text_utterances_CLIP.pickle'

    chain_count = 0
    utterance_count = 0

    for img_file in sorted(data):

        img_id = str(int(img_file.split('/')[1].split('.')[0].split('_')[2]))
        # img path in the form of 'person_bed/COCO_train2014_000000318646.jpg'
        # but also like 'bowl_dining_table/COCO_train2014_000000086285.jpg'

        chains4img = data[img_file]

        for game_id in sorted(chains4img):

            chain_data = chains4img[game_id]

            utt_ids = []  # pointers for the utterances in this chain
            utterance_lengths = []  # lengths of utterances in this chain

            for m in range(len(chain_data)):

                utterance_data = chain_data[m]
                message = utterance_data['Message_Text']
                message_nr = utterance_data['Message_Nr']
                round_nr = utterance_data['Round_Nr']

                tokenized_message = tweet_tokenizer.tokenize(message)

                # - / * AND SO ON punctuation marks are they in the dataset? vocab of bert?
                # INCLUDES * AND STUFF FOR CENSORING

                speaker = utterance_data['Message_Speaker']

                # visual context is the round images of the person who uttered the message
                if speaker == 'A':

                    visual_context = utterance_data['Round_Images_A']

                elif speaker == 'B':

                    visual_context = utterance_data['Round_Images_B']

                visual_context_ids = []

                for v in visual_context:

                    v_id = str(int(v.split('/')[1].split('.')[0].split('_')[2]))

                    visual_context_ids.append(v_id)

                visual_context_ids = sorted(visual_context_ids)  # SORTED VISUAL CONTEXT

                utt_length = len(tokenized_message) + 2  # WARNING!! ALREADY INCLUDING sos eos into the length
                # utterance information

                # WARNING THIS IS FOR CLIP and CLIPSCORE
                # SO I DON'T NEED TO USE THE tokenized_message
                # as the utterance, unlike PB
                # CLIP can tokenize the message itself
                # and we are not using the utterance lengths anyway
                utterance = {'utterance': message, 'image_set': visual_context_ids,
                             'target': [visual_context_ids.index(img_id)], 'length': utt_length, 'game_id': game_id,
                             'round_nr': round_nr, 'message_nr': message_nr}

                utterance_dataset[(game_id, round_nr, message_nr, img_id)] = utterance # add to the full dataset

                utterance_lengths.append(utt_length)
                utt_ids.append((game_id, round_nr, message_nr, img_id))
                utterance_count += 1

                if utterance_count % 500 == 0:
                    print(utterance_count)

            # chain information
            chain = {'game_id': game_id, 'chain_id': chain_count, 'utterances': utt_ids, 'target': img_id,
                     'lengths': utterance_lengths}  # utterance lengths

            chain_dataset.append(chain)
            chain_count += 1

    # dump the text versions of the chains and utterances

    with open(chains_path, 'w') as f:
        json.dump(chain_dataset, f)

    with open(utterances_path, 'wb') as f:
        pickle.dump(utterance_dataset, f)


with open('data/gold.json', 'r') as f:
    gold = json.load(f)

print(len(gold))

print('processing gold...')
process_data(gold, 'gold', min_freq)