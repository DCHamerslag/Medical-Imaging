

from utils.dataset import AIROGSLiteDataset, Rescale, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from training.models import get_model
from utils.parser import parse_args
from training.trainer import Trainer

import wandb
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
import toml
from utils.paths import ROOT


def main(args):
    print(args)
    if args.logging:
        wandb.init(project="test", entity="dchamerslag")
        wandb.config.update(args)

    transform = transforms.Compose([
        Rescale((1000, 1000)),
        ToTensor()
    ])
    training_config = {}
    dataset = AIROGSLiteDataset(transform)
    if args.device=="cuda":
        print("Using device: ", torch.cuda.get_device_name(0))
        device = torch.device(0)
    else:
        print("Using device: cpu")
        device = torch.device('cpu')
    model = get_model(args.model).to(device)

    plist = [
            {'params': model.layer4.parameters(), 'lr': 1e-5},
            {'params': model.last_linear.parameters(), 'lr': 5e-3}
            ]

    training_config['dataloader'] = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    training_config['model'] = model
    training_config['optimizer'] = optim.Adam(plist, lr=0.001)
    training_config['scheduler'] = lr_scheduler.StepLR(training_config['optimizer'], step_size=10, gamma=0.1)
    training_config['dataset_size'] = len(dataset)
    training_config['num_epochs'] = args.num_epochs
    training_config['loss_fn'] = nn.BCEWithLogitsLoss()
    training_config['device'] = device
    training_config['logging'] = args.logging

 
    trainer = Trainer(training_config)
    model = trainer.train()

    torch.save(model.state_dict(), "model.bin")

if __name__ == "__main__":  
    
    args = parse_args()
    main(args)