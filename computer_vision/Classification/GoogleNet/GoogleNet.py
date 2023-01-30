import torch
import torch.nn as nn

def conv_1(in_dim, out_dim): #1x1 conv
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1),
        nn.ReLU()
    )
    return model


def conv_1_3(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 3, 1, 1),
        nn.ReLU()
    )
    return model


def conv_1_5(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 5, 1, 2),
        nn.ReLU()
    )
    return model


def max_3_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(in_dim, out_dim, 1, 1),
        nn.ReLU()
    )
    return model

class Inception_module(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool_dim):
        super().__init__()

        self.conv_1 = conv_1(in_dim, out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3)
        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5)
        self.max_3_1 = max_3_1(in_dim, pool_dim)

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_1_3(x)
        out3 = self.conv_1_5(x)
        out4 = self.max_3_1(x)

        output = torch.cat([out1, out2, out3, out4], 1)
        return output

class Aux_classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(5, 3),
            nn.Conv2d(in_dim, 128, kernel_size = 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class GoogleNet(nn.Module):
    def __init__(self, base_dim = 64, Train_mode = True, num_classes = 10):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = base_dim, kernel_size = 7, stride = 2),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(in_channels = base_dim, out_channels = base_dim * 3, kernel_size = 3, stride = 1),
            nn.MaxPool2d(3, 2, 1)
        )

        self.inception3a = Inception_module(base_dim *3, 64, 96, 128, 16, 32, 32)
        self.LR = nn.LocalResponseNorm(2)
        self.inception3b = Inception_module(base_dim * 4, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)

        self.inception4a = Inception_module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_module(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(3, 2, 1)

        self.inception5a = Inception_module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_module(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(7, 1)

        self.dropout = nn.Dropout2d(0.4)
        self.fc_1 = nn.Linear(1024, num_classes)

        if Train_mode:
            self.aux1 = Aux_classifier(512, num_classes)
            self.aux2 = Aux_classifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.inception3a(x)
        x = self.LR(x)
        x = self.inception3b(x)
        x = self.LR(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.LR(x)

        if self.aux1 is not None:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.LR(x)
        x = self.inception4c(x)
        x = self.LR(x)
        x = self.inception4d(x)

        if self.aux2 is not None:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.LR(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.LR(x)
        x = self.inception5b(x)
        x = self.LR(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc_1(x)

        if self.aux1 is not None:
            return [x, aux1, aux2]
        else:
            return x

input = torch.rand(3, 1, 224, 224)
model = GoogleNet(Train_mode = True)
output, aux1, aux2 = model(input)
print(output.shape)
print(aux1.shape)
print(aux2.shape)







