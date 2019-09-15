
import torch.nn as nn
import torchvision


res_net = torchvision.models.resnet101(pretrained=True)

for p in res_net.children():
    print(p)

modules = list(res_net.children())[:-2]
res_net = nn.Sequential(*modules)
