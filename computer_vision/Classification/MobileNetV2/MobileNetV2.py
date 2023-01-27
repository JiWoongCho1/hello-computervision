import torch
import torch.nn as nn
from torchsummary import summary

def depthwise_conv(in_channels, stride = 1):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, stride =1, groups = in_channels, bias = False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6()
    )
    return layer

def conv1x1(in_channels, out_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0, stride= 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )
    return layer

def conv3x3(in_channels, out_channels, stride):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride= 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )
    return layer


class Inverted_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(conv1x1(in_channels, hidden_dim))
        layers.extend([depthwise_conv(hidden_dim, stride = stride), conv1x1(hidden_dim, out_channels)])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()

        self.configs = [
            #expand_ratio / channels / number / stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(in_channels, 32, stride = 2) #intial layer

        layers = []

        input_channels = 32 #same to output channels of intial layer

        for t, c, n, s in self.configs:
            for i in range(n): #iterate n(th)
                stride =s if i ==0 else 1 #after first iter, stride = 1
                layers.append(Inverted_Block(in_channels = input_channels, out_channels = c, expand_ratio = t, stride = stride))
                input_channels = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channels, 1280)

        self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x

input = torch.rand(3, 1, 224, 224)
model = MobileNetV2(in_channels = 1, num_classes = 10)
output = model(input)
print(summary(model, (1, 224, 224)))
print(output.shape)

