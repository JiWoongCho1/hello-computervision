import torch
import torch.nn as nn


def weights_init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        torch.nn.init.zeros(m.bias)
        print("nn.Linear init success")
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        print("nn.Conv2d init success")

def weights_init_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.zeros(m.bias)
        print("nn.Linear init success")
    elif type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        print("nn.Conv2d init success")

#model = Lenet()
#model.apply(weights_init_xavier)
#model.apply(weights_init_normal)
