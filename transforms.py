import torch.nn.functional as F
import torch


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes):
        C, H, W = image.shape

        r = min(self.size / H, self.size / W)

        new_W = int(W * r)
        new_H = int(H * r)

        canvas = torch.zeros((C, self.size, self.size), dtype=image.dtype)

        resized_image = F.interpolate(
            image.unsqueeze(0),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        padding_x = int((self.size - new_W) / 2)
        padding_y = int((self.size - new_H) / 2)

        canvas[:, padding_y:padding_y+new_H,
               padding_x:padding_x+new_W] = resized_image

        new_boxes = []

        for box in boxes:
            x, y, w, h = box

            x *= r
            y *= r
            w *= r
            h *= r

            x += padding_x
            y += padding_y

            new_boxes.append([x, y, w, h])

        return canvas, new_boxes
