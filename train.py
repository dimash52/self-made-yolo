import torch
from torch.utils.data import DataLoader

from dataset import CocoDataset
from model import SelfMadeYolo, encode
from loss import total_loss
from commons import collate_fn


# -------------------
# CONFIG
# -------------------
S = 8
C = 80
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
# DATA
# -------------------
dataset = CocoDataset("D:/self-made-yolo/coco2017")

small_dataset = torch.utils.data.Subset(dataset, range(20))

loader = DataLoader(
    small_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# -------------------
# MODEL
# -------------------
model = SelfMadeYolo().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -------------------
# TRAIN LOOP
# -------------------
for epoch in range(EPOCHS):

    model.train()
    total_epoch_loss = 0

    for images, boxes_batch, labels_batch in loader:

        images = images.to(device)

        preds = model(images)
        # (B, S, S, 5 + C)

        targets = []

        for b in range(len(boxes_batch)):
            target = encode(
                boxes_batch[b],
                labels_batch[b],
                S,
                C
            )
            targets.append(target)

        targets = torch.stack(targets).to(device)

        loss = total_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_epoch_loss:.4f}")
