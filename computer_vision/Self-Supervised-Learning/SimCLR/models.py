import torch.nn as nn
import torchvision.models as models
import torch


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.resnet = { "resnet18": models.resnet18(pretrained = True, num_classes = out_dim),
                        "resnet50": models.resnet50(pretrained = True, num_classes = out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )

    def _get_basemodel(self, model_name):
        model = self.resnet[model_name]
        return model

    def forward(self, x):
        return self.backbone(x)



