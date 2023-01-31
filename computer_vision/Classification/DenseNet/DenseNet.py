import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_dim, growth_rate):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,4 * growth_rate, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(4*growth_rate),
            nn.ReLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size = 3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = torch.cat([self.shortcut(x), self.residual(x)], 1)
        return x


class Transition(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.AvgPool2d(2, stride = 2)
        )
    def forward(self, x):
        return self.down_sample(x)



class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate = 12, reduction = 0.5, num_classes = 10):
        super().__init__()

        self.growth_rate = growth_rate
        inner_channels = growth_rate * 2

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, inner_channels, kernel_size = 7, stride= 2, padding = 3),
            nn.MaxPool2d(3, 2, 1)
        )

        self.features = nn.Sequential()

        for i in range(len(num_blocks) -1):
            self.features.add_module('dense_block{}'.format(i), self.make_dense_blocks(num_blocks[i], inner_channels))
            inner_channels += growth_rate * num_blocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(num_blocks)-1), self.make_dense_blocks(num_blocks[len(num_blocks)-1], inner_channels))
        inner_channels += growth_rate * num_blocks[len(num_blocks)-1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(inner_channels, num_classes)

    def make_dense_blocks(self, num_blocks, inner_channels):
        dense_block = nn.Sequential()

        for i in range(num_blocks):
            dense_block.add_module('bottle_neck{}'.format(i), BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        x = self.initial_block(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

input = torch.rand(5, 3, 64, 64)
model = DenseNet([6, 12, 24, 6], growth_rate = 12, reduction = 0.5, num_classes = 10)
output = model(input)
print(output.shape)


