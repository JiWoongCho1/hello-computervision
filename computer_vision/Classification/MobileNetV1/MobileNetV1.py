import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary


class DepthwiseSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding = 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, width_multiplier, num_classes):
        super().__init__()

        alpha = width_multiplier
        self.conv1 = CBR(3, int(32 * alpha), kernel_size = 3, stride =2, padding = 1)
        self.conv2 = DepthwiseSeperable(int(32 * alpha), int(64 * alpha), stride = 1)
        self.conv3 = nn.Sequential(
            DepthwiseSeperable(int(64 * alpha), int(128 * alpha), stride = 2),
            DepthwiseSeperable(int(128 * alpha), int(128 * alpha), stride = 1)
        )
        self.conv4 = nn.Sequential(
            DepthwiseSeperable(int(128 * alpha), int(256 * alpha), stride = 2),
            DepthwiseSeperable(int(256 * alpha), int(256 * alpha), stride = 1)
        )
        self.conv5 = nn.Sequential(
            DepthwiseSeperable(int(256 * alpha), int(512 * alpha), stride=2),
            DepthwiseSeperable(int(512 * alpha), int(512 * alpha), stride=1),
            DepthwiseSeperable(int(512 * alpha), int(512 * alpha), stride=1),
            DepthwiseSeperable(int(512 * alpha), int(512 * alpha), stride=1),
            DepthwiseSeperable(int(512 * alpha), int(512 * alpha), stride=1),
            DepthwiseSeperable(int(512 * alpha), int(512 * alpha), stride=1)
        )
        self.conv6 = nn.Sequential(
            DepthwiseSeperable(int(512 * alpha), int(1024 * alpha), stride = 2)
        )
        self.conv7 = nn.Sequential(
            DepthwiseSeperable(int(1024 * alpha), int(1024 * alpha), stride = 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = MobileNetV1(width_multiplier = 1, num_classes = 10)
input = torch.rand(5, 3, 224, 224)
output = model(input)
print(summary(model, (3, 224, 224)))
print(output.shape)


