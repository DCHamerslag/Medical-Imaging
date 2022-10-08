from typing import Dict, List

from cv2 import transform
from utils.dataset import AIROGSLiteDataset, Rescale, ToTensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
from training.models import get_model
from utils.parser import parse_args
from training.trainer import Trainer

import wandb
import torch
import torch.nn as nn

import toml
from utils.paths import ROOT

def main(args):
    print("Running with parameters: ", args)
    if args.logging:
        wandb.init(project="test-project", entity="airogs-project")
        wandb.config.update(args)

    device = find_device()

    transform = transforms.Compose([
        Rescale((args.rescale_w, args.rescale_h)),
        ToTensor()
    ])
    split = [13000, 2000]
    train_loader, test_loader = create_dataloaders(args, transform, split)

    model, optimizer, scheduler = get_model(args.model, args.model_name)
    model = model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    training_config = {}
    training_config['dataloader'] = train_loader
    training_config['test_dataloader'] = test_loader
    training_config['model'] = model
    training_config['optimizer'] = optimizer
    training_config['scheduler'] = scheduler
    training_config['dataset_size'] = split[0]
    training_config['num_epochs'] = args.num_epochs
    training_config['loss_fn'] = loss_fn
    training_config['device'] = device
    training_config['logging'] = args.logging
    training_config['batch_size'] = args.batch_size

    trainer = Trainer(training_config)
    model = trainer.train()

    if(args.model_name == ''):
        model_name = 'model.bin'
    else: 
        model_name = args.model_name
    torch.save(model.state_dict(), model_name)

def create_dataloaders(args: Dict, transform: transform, split: List):
    dataset = AIROGSLiteDataset(args, transform)
    dataset.shuffle()               ## instead of random_split(), we shuffle the list and then just split the data in train and test by index
    class_weights = [1, 1500/13500] ## this is more efficient for creating our sample weights, for the weightedrandomsampler
    sample_weights = [0] * len(dataset)
    for idx, (_,label) in enumerate(dataset.labels): ## we give each sample a weight of 1 for no glaucoma, or 1500/13500 for glaucoma
        sample_weights[idx] = class_weights[1] if label == 'NRG' else class_weights[0]
        
    train_sampler = WeightedRandomSampler(sample_weights[:split[0]], num_samples=split[0], replacement=True)
    test_sampler = WeightedRandomSampler(sample_weights[split[0]:], num_samples=split[1], replacement=True)
    #train_set, test_set = torch.utils.data.random_split(dataset, split)
    train_set = torch.utils.data.Subset(dataset,range(split[0]))
    test_set = torch.utils.data.Subset(dataset,range(split[0], split[0]+split[1]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=test_sampler)
    return train_loader, test_loader

def find_device() -> torch.DeviceObjType:
    if args.device=="cuda":
        print("Using device: ", torch.cuda.get_device_name(0))
        device = torch.device(0)
    else:
        print("Using device: cpu")
        device = torch.device('cpu')

    return device

if __name__ == "__main__":  
    args = parse_args()
    main(args)