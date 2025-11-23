from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as data

train_data = FashionMNIST('./data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([transforms.Resize(size=244),
                                                 transforms.ToTensor()]))

train_loader = data.DataLoader(train_data,   #数据加载进来
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.squeeze().numpy()
    class_label = train_data.classes
    print(class_label)
