
# Copied from `train` function in train_simple.py:L78
import yaml
import os
import sys
abs_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
device = 'cpu'
hyp = f'{abs_path}/data/hyps/hyp.scratch-low.yaml'

with open(hyp, errors="ignore") as f:
    hyp = yaml.safe_load(f)  # load hyps dict

sys.path.append(abs_path)

from models.yolo import Model
from utils.general import check_dataset

cfg = f'{abs_path}/models/yolov5n_kaist-rgbt.yaml'
data = f'{abs_path}/data/kaist-rgbt.yaml'
data_dict = check_dataset(data)

nc = int(data_dict["nc"])  # number of classes
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

import cv2
import os
import numpy as np
from PIL import Image

annFile = f'{abs_path}/datasets/kaist-rgbt/train/labels/set05_V000_I01219.txt'
lwirFile = annFile.replace('labels', 'images/lwir').replace('.txt', '.jpg')
visFile  = annFile.replace('labels', 'images/visible').replace('.txt', '.jpg')

# Read images
img_lwir = cv2.imread(lwirFile)
img_vis  = cv2.imread(visFile)

h, w = img_vis.shape[:2]

# Read labels
with open(annFile, 'r') as fp:
    labels = [x.split() for x in fp.read().strip().splitlines() if len(x)]

colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 255),
}

if len(labels):
    # convert normalized bbox to pixel coordinates
    labels = np.array(labels, dtype=np.float32)
    labels[:, (1, 3)] *= w
    labels[:, (2, 4)] *= h

    cls = labels[:, 0]

    xyxy = np.zeros((len(labels), 4))
    xyxy[:, :2] = labels[:, 1:3]
    xyxy[:, 2:] = labels[:, 1:3] + labels[:, 3:5]
    xyxy = xyxy.astype(np.int16)

    for c, bb in zip(cls, xyxy):
        color = colors[c]
        cv2.rectangle(img_lwir, bb[:2], bb[2:], color)
        cv2.rectangle(img_vis,  bb[:2], bb[2:], color)

images = np.concatenate([img_lwir, img_vis], axis=1)
Image.fromarray(images)


import os
# from path_utils import absolute_path
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, colorstr

imgsz = 640
batch_size = 1
single_cls = False
seed = 0

train_path = data_dict["train"]
val_path = data_dict["val"]
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

train_loader, dataset = create_dataloader(
    # train_path,
    val_path,
    imgsz,
    batch_size,
    gs,
    single_cls,
    hyp=hyp,
    augment=False,      # TODO: check if there is no bug when applying augmentation
    cache=None,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix=colorstr("train: "),
    shuffle=False,      # No shuffle for debugging
    seed=seed,
    rgbt_input=True,
)
from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.general import xywh2xyxy

# frame = 12112

# Get a minibatch
# for ii, (imgs, targets, paths, _) in enumerate(train_loader):
#     break

# Get a minibatch (fast)
save_annotation_dict ={}
image_info_list = []
annotation_info_list = []
for frame in range(2508):
    imgs, targets, paths, shapes, _ = LoadRGBTImagesAndLabels.collate_fn([dataset[frame]])

    

    idx = 0
    img_lwir = imgs[0][idx].numpy().transpose((1, 2, 0))
    img_vis  = imgs[1][idx].numpy().transpose((1, 2, 0))
    h, w = img_vis.shape[:2]

    labels = targets.numpy()

    colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 255),
    }

    if len(labels):
        labels = labels[labels[:, 0] == idx, 1:]

        # convert normalized bbox to pixel coordinates
        labels = np.array(labels, dtype=np.float32)
        labels[:, (1, 3)] *= w
        labels[:, (2, 4)] *= h

        image_info = {}
        image_info['id'] = frame
        image_info['im_name'] = paths[0].split('/')[-1].rstrip('.jpg')
        image_info['height'] = h
        image_info['weight'] = w
        image_info_list.append(image_info)

        annotation_info = {}
        annotation_info['id'] = idx
        annotation_info['image_id'] = frame
        annotation_info['category_id'] = int(labels.tolist()[0][0])
        annotation_info['bbox'] = labels.tolist()[0][1:]
        annotation_info_list.append(annotation_info)
        
        idx += 1

        cls = labels[:, 0]

        xyxy = xywh2xyxy(labels[:, 1:5])
        xyxy = xyxy.astype(np.int16)

        img_lwir = np.ascontiguousarray(img_lwir)
        img_vis = np.ascontiguousarray(img_vis)

save_annotation_dict['images'] = image_info_list
save_annotation_dict['annotations'] = annotation_info_list

import json
with open(f'{abs_path}/KAIST_annotation.json', 'w') as f:
    json.dump(save_annotation_dict, f)

    #     for c, bb in zip(cls, xyxy):
    #         color = colors[c]
    #         cv2.rectangle(img_lwir, bb[:2], bb[2:], color)
    #         cv2.rectangle(img_vis,  bb[:2], bb[2:], color)

    # images = np.concatenate([img_lwir, img_vis], axis=1)
    # print(paths[idx])
    # Image.fromarray(images)


    # "images": [
    #     {
    #         "id": 0,
    #         "im_name": "set06/V000/I00019",
    #         "height": 512,
    #         "width": 640
    #     },














# label_path = f'{absolute_path()}/train/labels'




# with open(f'{absolute_path()}/val-split-04.txt', 'r') as f:
#     val_files = [line.strip().split('/')[-1] for line in f.readlines()]

# for val_file in val_files:
#     for label_file in os.listdir(label_path):
#         val_file = val_file.rstrip('.jpg')
#         label_file = label_file.rstrip('.txt')
#         if val_file == label_file:
#             with open(f'{label_path}/{label_file}.txt') as t:
#                 annotations = [line.strip() for line in t.readlines()]