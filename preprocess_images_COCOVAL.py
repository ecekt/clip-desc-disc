import torch
import clip
from PIL import Image

import json
from collections import defaultdict
import pickle

with open('annotations_trainval2017/annotations/captions_val2017.json', 'r') as f:
    val_coco = json.load(f)

coco_id2file = defaultdict()

for im in val_coco['images']:

    coco_id2file[im['id']] = '/home/ece/Desktop/phd_repos/mm_matching/val2017/'\
                             + im['file_name']

del val_coco

clip_preprocessed_images = defaultdict()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

count_img = 0

print('len imgs', len(coco_id2file))  # 5000

for img in coco_id2file:

    count_img += 1
    if count_img % 100 == 0:
        print(count_img)

    image_path = coco_id2file[img]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    clip_preprocessed_images[img] = image

with open('data/clip_preprocessed_images_COCOVAL.pickle', 'wb') as f:
    pickle.dump(clip_preprocessed_images, f)
