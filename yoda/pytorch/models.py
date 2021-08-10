import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # 방법 1
# class mnist_model(torch.nn.Module):
#     def __init__(self) -> None:
#         super(mnist_model, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(64 * 7 * 7, 64)
#         self.fc2 = nn.Linear(64, 10)
    
#     def forward(self, x):

#         out = F.relu(self.conv1(x))
#         out = self.pool1(out)
#         out = F.relu(self.conv2(out))
#         out = self.pool2(out)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))

#         return out

# # 방법 2
# class mnist_model(torch.nn.Module):
#     def __init__(self) -> None:
#         super(mnist_model, self).__init__()

#         self.cnn = torch.nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#         )

#         self.fc = torch.nn.Sequential(
#             nn.Linear(64 * 7 * 7, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10)
#         )
    
#     def forward(self, x):

#         out = self.cnn(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return out

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
