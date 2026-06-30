import torch
import torch.nn as nn
import torch.nn.functional as F


def encode(boxes, labels, S, C):

    target = torch.zeros((S, S, 5 + C))

    for box, label in zip(boxes, labels):

        cx, cy, w, h = box

        i = int(cy * S)
        j = int(cx * S)

        # safety clamp
        i = max(0, min(S - 1, i))
        j = max(0, min(S - 1, j))

        tx = cx * S - j
        ty = cy * S - i

        target[i, j, 0] = tx
        target[i, j, 1] = ty
        target[i, j, 2] = w
        target[i, j, 3] = h
        target[i, j, 4] = 1.0

        target[i, j, 5 + label] = 1.0

    return target


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 1/2


            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 1/4


            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # 1/8


            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.net(x)


class YoloHead(nn.Module):
    def __init__(self, in_channels, num_boxes, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.num_boxes = num_boxes

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_boxes*(5 + num_classes),
            kernel_size=1
        )

    def forward(self, x):

        x = self.conv(x)

        B, _, S, _ = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class SelfMadeYolo(nn.Module):
    def __init__(self, num_boxes=2, num_classes=20):
        super().__init__()

        self.backbone = BackBone()
        self.head = YoloHead(
            in_channels=256, num_boxes=num_boxes, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        preds = self.head(features)
        return preds


class YoloDecoder():
    def __init__(self, S, num_boxes, num_classes):

        self.S = S
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def decode(self, preds, threshold):
        '''
        decoding input image into
        probability and class_id
        '''
        B = preds.shape[0]

        preds = preds.view(
            B,
            self.S,
            self.S,
            self.num_boxes,
            5 + self.num_classes
        )

        DEVICE = preds.device

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.S, device=DEVICE),
            torch.arange(self.S, device=DEVICE),
            indexing='ij'
        )

        x_grid = x_grid.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        box_xy = torch.sigmoid(preds[..., 0:2])
        box_wh = preds[..., 2:4]

        obj = torch.sigmoid(preds[..., 4])
        cls = torch.softmax(preds[..., 5], dim=-1)

        # converting into image dimension (scaling)

        box_xy = (box_xy + torch.cat([x_grid, y_grid], dim=-1)) / self.S
        box_wh = torch.exp(box_wh) / self.S

        scores = obj.unsqueeze(-1) * cls

        detections = []

        for b in range(B):
            batch_boxes = []
            batch_scores = []
            batch_classes = []

            for i in range(self.S):
                for j in range(self.S):
                    for k in range(self.num_boxes):

                        score, class_id = torch.max(scores[b, i, j, k], dim=-1)

                        if score < threshold:
                            continue

                        coord_x, coord_y = box_xy[b, i, j, k]
                        w, h = box_wh[b, i, j, k]

                        batch_boxes.append([coord_x, coord_y, w, h])
                        batch_scores.append(score)
                        batch_classes.append(class_id.item())

            detections.append({
                'boxes': torch.tensor(batch_boxes),
                'score': torch.tensor(batch_scores),
                'classes': torch.tensor(batch_classes),
            })

        return detections
