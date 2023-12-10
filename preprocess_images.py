import torch
import clip
from PIL import Image

import json
from collections import defaultdict
import pickle

clip_preprocessed_images = defaultdict()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('coco_id2file.json', 'r') as f:
    coco_id2file = json.load(f)

count_img = 0

print('len imgs', len(coco_id2file))  # 358

for img in coco_id2file:

    count_img += 1
    if count_img % 10 == 0:
        print(count_img)

    image_path = 'photobook_coco_images/images/' + coco_id2file[img]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    clip_preprocessed_images[img] = image

with open('data/clip_preprocessed_images.pickle', 'wb') as f:
    pickle.dump(clip_preprocessed_images, f)