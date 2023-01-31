import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride= 1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, stride= 1 , padding = 1, bias = False),
            nn.BatchNorm2d(out_dim)
        )
        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_dim)
            )
    def forward(self, x):
        x = self.layer(x)
        x = self.shortcut(x) + x
        x = torch.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, ):
        super().__init__()


        self.in_dim = 64

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layers(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self.make_layers(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self.make_layers(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self.make_layers(block, 512, num_blocks[3], stride = 2)
        self.classifier = nn.Linear(512, num_classes)

    def make_layers(self, block, out_dim, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_dim, out_dim, stride))
            self.in_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

input = torch.rand(5, 3, 32,32)
model = ResNet(BasicBlock, [2,2,2,2], num_classes = 10)
output = model(input)
print(output.shape)