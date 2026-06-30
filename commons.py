import torch


def collate_fn(batch):

    images = []
    boxes = []
    labels = []

    for b in batch:
        images.append(b["image"])
        boxes.append(b["boxes"])
        labels.append(b["labels"])

    images = torch.stack(images)

    return images, boxes, labels
