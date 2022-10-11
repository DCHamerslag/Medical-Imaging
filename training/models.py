from argparse import Namespace
from typing import Any, Tuple
import torch.nn as nn
import pretrainedmodels as pm
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import load
import torch
import pytorch_pretrained_vit as ptv
from torchvision import models


def get_model(args: Namespace):
    name = args.model
    if name=="ResNet50":
        return get_pretrained_resnet50()
    if name=="ViT-pretrained":
        return get_pretrained_ViT(args)
    if name=="ResNet50-untrained":
        return get_resnet50_untrained()

def get_pretrained_resnet50():
    ''' Return ResNet50 with hardcoded optimizer and scheduler. '''
    model = pm.__dict__["resnet50"](pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=128),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=128, out_features=2),
    )

    plist = [
            {'params': model.layer4.parameters(), 'lr': 1e-5},
            {'params': model.last_linear.parameters(), 'lr': 5e-3}
    ]
    optimizer = optim.Adam(plist, lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return model, optimizer, scheduler

def get_resnet50_untrained():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return model, optimizer, scheduler

def get_pretrained_ViT(args: Namespace):
    model_name = args.model_name
    if model_name != '':
        model = ptv.ViT('B_16', pretrained=False, num_classes=2)
        print(args.device)
        if args.device == 'cpu':
            model.load_state_dict(load(args.model_dir + "/" + model_name, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(load(args.model_dir + "/" + model_name))
    else:
        model = ptv.ViT('B_16', pretrained=True, num_classes=2) ## L_32 is the best model, but B_16 best size/performance.
    optimizer = optim.Adam(model.parameters(), lr=0.001) ## not sure how to call specific blocks, ViT has blocks instead of layers.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)    
    return model, optimizer, scheduler

