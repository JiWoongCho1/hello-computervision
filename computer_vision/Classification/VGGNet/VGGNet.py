import torch
import torch.nn as nn

layer_config = {'VGG16' :
                [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',512, 512, 512, 'M'],
                'VGG19' :
                [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',512, 512, 512, 512, 'M']}

def make_layers(config):
    layers = []
    in_channels = 3

    for value in  config:
        if value == 'M':
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        else:
            layers.append(nn.Conv2d(in_channels = in_channels, out_channels = value, kernel_size = 3, stride = 1, padding = 1))
            layers.append(nn.ReLU())
            in_channels = value
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, config, num_classes = 10):
        super().__init__()

        self.feature_extractor = make_layers(config)

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # or x.view(-1, 7*7*512)
        x = self.classifier(x)
        return x

def VGG16_(config):
    return VGG(config['VGG16'])

def VGG19_(config):
    return VGG(config['VGG19'])

class VGGEnsemble(nn.Module):
    def __init__(self, modelA, modelB, num_classes):
        super().__init__()

        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):

        x1 = self.modelA(x)
        x2 = self.modelB(x)

        output = x1 + x2
        output = nn.Softmax(dim = 1)(output)
        return output

VGG16 = VGG16_(layer_config)
VGG19 = VGG19_(layer_config)

input = torch.rand(5, 3, 224, 224)
ensemble_model = VGGEnsemble(VGG16, VGG19, 10)
output = ensemble_model(input)
print(output.shape)
