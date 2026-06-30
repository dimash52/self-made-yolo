from model import SelfMadeYolo
import torch

model = SelfMadeYolo()
x = torch.randn(2, 3, 256, 256)
output = model(x)
print(output.shape)
