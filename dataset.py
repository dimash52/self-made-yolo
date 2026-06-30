from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path
from collections import OrderedDict, defaultdict
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import cv2
import json


class CocoDataset(Dataset):
    def __init__(self, datapath, augmentations=False):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        self.datapath = Path(datapath)
        self.augmentations = augmentations

        self.annotations = defaultdict(list)

        ann_file = self.datapath / "annotations/instances_train2017.json"

        with open(ann_file) as f:
            data = json.load(f)

            self.images = {img['id']: img for img in data["images"]}

            for ann in data["annotations"]:
                self.annotations[ann["image_id"]].append(ann)

            # annotations also might be empty
            self.categories = {c["id"]: c["name"] for c in data["categories"]}

        self.images_idx = list(self.images.keys())

        self.cat2idx = {cat_id: i for i,
                        cat_id in enumerate(self.categories.keys())}

    def __getitem__(self, idx):
        image_id = self.images_idx[idx]
        image_info = self.images[image_id]

        img_path = self.datapath / 'train2017' / image_info["file_name"]

        img_w = image_info['width']
        img_h = image_info['height']

        image = Image.open(img_path).convert('RGB')

        anns = self.annotations[image_id]

        boxes = []
        labels = []

        for a in anns:
            x, y, w, h = a['bbox']

            cx = (x + w) / 2
            cy = (y + h) / 2

            cx /= img_w
            cy /= img_h
            w /= img_w
            h /= img_h

            boxes.append([cx, cy, w, h])
            labels.append(self.cat2idx[a['category_id']])

        return {
            "image": self.transform(image),
            "boxes": boxes,
            "labels": labels
        }

    def __len__(self):
        return len(self.images_idx)


if __name__ == '__main__':
    # s = CocoDataset('D:\self-made-yolo\coco2017')
    dataset = CocoDataset('D:\self-made-yolo\coco2017')

    img, boxes, labels = dataset[0]
