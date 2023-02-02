import torch
import torch.nn as nn

class SeperableConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.seperable = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride= 1, padding = 0, bias = False )
        )
    def forward(self, x):
        x = self.seperable(x)
        return x


class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding =  1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 0, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            SeperableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeperableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, 0),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            SeperableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeperableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, 0),
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            SeperableConv(256, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1, 2, 0),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + self.conv2_shortcut(x)
        x = self.conv3(x) + self.conv3_shortcut(x)
        x = self.conv4(x) + self.conv4_shortcut(x)

        return x


class MiddleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728)
        )
        self.conv_shortcut = nn.Sequential()
        
    def forward(self, x):
        x = self.conv_residual(x) + self.conv_shortcut(x)
        return x
    
class ExitFlow(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            SeperableConv(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SeperableConv(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.conv1_shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, 2, 0),
            nn.BatchNorm2d(1024)
        )
        
        self.conv2 = nn.Sequential(
            SeperableConv(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeperableConv(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.conv1(x) + self.conv1_shortcut(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        self.entry = EntryFlow()
        self.middle = self.make_middle_flow()
        self.exit = ExitFlow()
        self.classifier = nn.Linear(2048, num_classes)

    def make_middle_flow(self):
        middle = nn.Sequential()
        for i in range(8):
            middle.add_module('middle_block{}'.format(i), MiddleFlow())
        return middle

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

input = torch.rand(2, 3, 299, 299)
model = Xception()
output = model(input)
print(output.shape)
