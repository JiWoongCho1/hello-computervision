import torch
import torch.nn as nn
import torch.optim as optim
import torchvision



class CRNM(nn.Module):
    def __init__(self, in_channels_, out_channels_, kernel_size_, stride_, padding_, bias_ = False):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_, out_channels=out_channels_, kernel_size = kernel_size_, stride = stride_, padding = padding_, bias = bias_),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5, k = 2, beta = 0.75, alpha = 0.0001),
            nn.MaxPool2d(kernel_size = 3, stride =2)
        )
    def forward(self, x):
        return self.layer(x)



class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes = 10):
        super().__init__()

        self.layer1 = CRNM(1, 96, 11, 4, 2, False)
        self.layer2 = CRNM(96, 256, 5, 1, 2, False)
        self.conv3 = nn.Conv2d(256, 384, 3, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(384, 384, 3, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(384, 256, 3, padding = 1, bias = False)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

model = AlexNet(1, 10)
input = torch.rand(5, 1, 224, 224)
output = model(input)
print(output.shape)
