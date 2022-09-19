import torch.nn as nn
import pretrainedmodels as pm

def get_model(name: str):
    if name=="ResNet50":
        return get_pretrained_resnet50()

def get_pretrained_resnet50():

    model = pm.__dict__["resnet50"](pretrained='imagenet')

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=2),
    )
    return model