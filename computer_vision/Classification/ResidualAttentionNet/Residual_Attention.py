import torch
import torch.nn as nn
import torch.nn.functional as F

class PreactResidual(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()

        bottleneck_dim = int(in_dim / 4)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, bottleneck_dim, 1,stride = stride, padding = 0),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, stride=stride, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim, out_dim, 1, stride=stride, padding=0),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Conv2d(in_dim, out_dim, 1, stride = stride)

    def forward(self, x):
        x = self.residual(x) + self.shortcut(x)
        return x


class AttentionModule1(nn.Module):
    def __init__(self, in_dim, out_dim, p = 1, t = 2, r = 1):
        super().__init__()

        assert in_dim == out_dim

        self.stem = self.make_residual(in_dim, out_dim, p)
        self.trunk = self.make_residual(in_dim, out_dim, t)

        self.soft_resdown1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown2 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown3 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown4 = self.make_residual(in_dim, out_dim, r)

        self.soft_resup1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup2 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup3 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup4 = self.make_residual(in_dim, out_dim, r)

        self.shortcut_short = PreactResidual(in_dim, out_dim, 1)
        self.shortcut_long = PreactResidual(in_dim, out_dim, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride=  1, padding = 0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )#select which location should be focused

        self.last = self.make_residual(in_dim, out_dim, p)
    def make_residual(self, in_dim, out_dim, p):
        layers = []
        for _ in range(p):
            layers.append(PreactResidual(in_dim, out_dim, 1))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)
        input_size = (x.size(2), x.size(3))

        x_trunk = self.trunk(x)

        x_soft = F.max_pool2d(x, 3, 2, 1)
        x_soft = self.soft_resdown1(x_soft)

        shape1 = ((x_soft.size(2), x_soft.size(3)))
        shortcut_long = self.shortcut_long(x_soft)

        x_soft = F.max_pool2d(x_soft, 3, 2, 1)
        x_soft = self.soft_resdown2(x_soft)

        shape2 = ((x_soft.size(2), x_soft.size(3)))
        shortcut_short = self.shortcut_short(x_soft)

        x_soft = F.max_pool2d(x_soft, 3, 2, 1)
        x_soft = self.soft_resdown3(x_soft)

        x_soft = self.soft_resdown4(x_soft)
        x_soft = self.soft_resup1(x_soft)

        x_soft = self.soft_resup2(x_soft)
        x_soft = F.interpolate(x_soft, size = shape2)
        x_soft += shortcut_short

        x_soft = self.soft_resup3(x_soft)
        x_soft = F.interpolate(x_soft, size = shape1)
        x_soft = shortcut_long

        x_soft = self.soft_resup4(x_soft)
        x_soft = F.interpolate(x_soft, size = input_size)

        x_soft = self.sigmoid(x_soft)
        x = (1 + x_soft) * x_trunk
        x = self.last(x)

        return x


class AttentionModule2(nn.Module):
    def __init__(self, in_dim, out_dim, p=1, t=2, r=1):
        super().__init__()

        assert in_dim == out_dim

        self.stem = self.make_residual(in_dim, out_dim, p)
        self.trunk = self.make_residual(in_dim, out_dim, t)

        self.soft_resdown1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown2 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown3 = self.make_residual(in_dim, out_dim, r)


        self.soft_resup1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup2 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup3 = self.make_residual(in_dim, out_dim, r)

        self.shortcut = PreactResidual(in_dim, out_dim, 1)

        self.sigmoid= nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.last = self.make_residual(in_dim, out_dim, p)

    def make_residual(self, in_dim, out_dim, p):
        layers = []
        for _ in range(p):
            layers.append(PreactResidual(in_dim, out_dim, 1))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)
        input_size = (x.size(2), x.size(3))

        x_trunk = self.trunk(x)

        x_soft = F.max_pool2d(x, 3, 2, 1)
        x_soft = self.soft_resdown1(x_soft)

        shape1 = ((x_soft.size(2), x_soft.size(3)))
        shortcut= self.shortcut(x_soft)

        x_soft = F.max_pool2d(x_soft, 3, 2, 1)
        x_soft = self.soft_resdown2(x_soft)

        x_soft = self.soft_resdown3(x_soft)
        x_soft = self.soft_resup1(x_soft)

        x_soft = self.soft_resup2(x_soft)
        x_soft = F.interpolate(x_soft, size=shape1)
        x_soft += shortcut

        x_soft = self.soft_resup3(x_soft)
        x_soft = F.interpolate(x_soft, size=input_size)

        x_soft = self.sigmoid(x_soft)
        x = (1 + x_soft) * x_trunk
        x = self.last(x)

        return x


class AttentionModule3(nn.Module):
    def __init__(self, in_dim, out_dim, p=1, t=2, r=1):
        super().__init__()

        assert in_dim == out_dim

        self.stem = self.make_residual(in_dim, out_dim, p)
        self.trunk = self.make_residual(in_dim, out_dim, t)

        self.soft_resdown1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resdown2 = self.make_residual(in_dim, out_dim, r)

        self.soft_resup1 = self.make_residual(in_dim, out_dim, r)
        self.soft_resup2 = self.make_residual(in_dim, out_dim, r)

        self.shortcut = PreactResidual(in_dim, out_dim, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.last = self.make_residual(in_dim, out_dim, p)

    def make_residual(self, in_dim, out_dim, p):
        layers = []
        for _ in range(p):
            layers.append(PreactResidual(in_dim, out_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        input_size = (x.size(2), x.size(3))

        x_trunk = self.trunk(x)

        x_soft = F.max_pool2d(x, 3, 2, 1)
        x_soft = self.soft_resdown1(x_soft)

        x_soft = self.soft_resdown2(x_soft)
        x_soft = self.soft_resup1(x_soft)

        x_soft = self.soft_resup2(x_soft)
        x_soft = F.interpolate(x_soft, size=input_size)

        x_soft = self.sigmoid(x_soft)
        x = (1 + x_soft) * x_trunk
        x = self.last(x)

        return x



class AttentionNet(nn.Module):
    def __init__(self, num_blocks, num_classes = 10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.stage1 = self.make_stage(64, 256, num_blocks[0], AttentionModule1, 1)
        self.stage2 = self.make_stage(256, 512, num_blocks[1], AttentionModule2, 1)
        self.stage3 = self.make_stage(512, 1024, num_blocks[2], AttentionModule3, 1)

        self.stage4 = nn.Sequential(
            PreactResidual(1024, 2048, 1),
            PreactResidual(2048, 2048, 1),
            PreactResidual(2048, 2048, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def make_stage(self, in_dim, out_dim, num_blocks, block, stride):
        stage = []
        stage.append(PreactResidual(in_dim, out_dim, stride))

        for i in range(num_blocks):
            stage.append(block(out_dim, out_dim))
        return nn.Sequential(*stage)


    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

input = torch.rand(2, 3, 224, 224)
model = AttentionNet([1,1,1])
output = model(input)
print(output.shape)
