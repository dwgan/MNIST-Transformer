import torch
import time
from torch import optim
from test import test
from train import train
from data_loader import train_loader, test_loader
from torch.optim.lr_scheduler import MultiStepLR
from model import ViT

torch.manual_seed(42)

data_path = '../dataset/mnist'
batch_size = 64
epochs = 20
learning_rate = 1e-3

start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。'''
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

for epoch in range(1, epochs + 1):
    print('Epoch:', epoch)
    train(model, device, optimizer, scheduler, train_loader(data_path, batch_size))
    test(model, device, test_loader(data_path, batch_size))

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')