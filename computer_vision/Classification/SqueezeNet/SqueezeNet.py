import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_dim, squeeze_dim, expand1x1_dim, expand3x3_dim):
        super().__init__()

        self.in_dim = in_dim
        self.squeeze = nn.Conv2d(in_dim, squeeze_dim, kernel_size = 1)
        self.expand1x1 = nn.Conv2d(squeeze_dim, expand1x1_dim, kernel_size = 1)
        self.expand3x3 = nn.Conv2d(squeeze_dim, expand3x3_dim, kernel_size=3, padding = 1)

    def forward(self, x):
        x = torch.relu(self.squeeze(x))
        x = torch.cat([torch.relu(self.expand1x1(x)), torch.relu(self.expand3x3(x))], 1)
        return x

class SqueezeNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 7, stride= 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, ceil_mode = True)
        )
        self.middle_block = nn.Sequential(
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(3, 2, ceil_mode = True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(3, 2, ceil_mode = True),
            Fire(512, 64, 256, 256)
        )
        self.final_conv = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size = 1)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x):
        x = self.initial_block(x)
        x = self.middle_block(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

input = torch.rand(5, 3, 224, 224)
model = SqueezeNet()
output = model(input)
print(output.shape)