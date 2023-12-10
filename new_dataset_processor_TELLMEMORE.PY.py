import os
import csv
import json

import clip
import torch

import pickle
from PIL import Image

from collections import defaultdict

clip_preprocessed_images = defaultdict()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data_path = '/home/ece/Desktop/phd_repos/2021datasets/image-description-sequences-master/data/'
vis_path = '/home/ece/Desktop/phd_repos/2021datasets/image-description-sequences-master/data/image_data/ADE20K_2016_07_26/images/'

with open(data_path + 'splits.json', 'r') as f:
    splits = json.load(f)

# these are single captions per image, 411
with open(data_path + 'captions.csv', "r") as f:
    reader = csv.reader(f, delimiter='\t')

    header = True
    count = 0

    tmm_captions = dict()

    for l in reader:

        if header:
            #print(l)  # ['', 'caption_id', 'image_id', 'caption']
            header = False

        else:
            #print(l)
            count += 1

            image_id = l[2]
            caption = l[3]

            assert image_id not in tmm_captions
            tmm_captions[image_id] = caption

    print(count)
    print(len(tmm_captions))


# these are the descriptions where the subjects TOLD US MORE, 5702

with open(data_path + 'sequences.csv', "r") as f:
    reader = csv.reader(f, delimiter='\t')

    header = True
    count = 0

    tmm_seqs = []
    image_paths = []

    for l in reader:

        if header:
            #print(l)  # ['', 'seq_id', 'image_id', 'image_path', 'image_cat', 'image_subcat', 'd1', 'd2', 'd3', 'd4', 'd5']
            header = False

        else:
            #print(l)
            count += 1

            if count % 100 == 0:
                print(count)

            image_id = l[2]
            image_path = l[3]
            sequence = l[6:]

            tmm_seqs.append({"image": vis_path + image_path, "sequence": sequence})

            image_paths.append(image_path)
            # if image_id not in clip_preprocessed_images:
            #
            #     image_path = vis_path + image_path
            #
            #     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            #
            #     clip_preprocessed_images[image_path] = image


    print(count)
    print(len(tmm_seqs))


# with open('data/clip_preprocessed_images_TELLMEMORE.pickle', 'wb') as f:
#     pickle.dump(clip_preprocessed_images, f)

with open('data/full_TELLMEMORE.json', 'w') as f:
    json.dump(tmm_seqs, f)
