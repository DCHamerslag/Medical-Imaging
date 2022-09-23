

from utils.dataset import AIROGSLiteDataset, Rescale, ToTensor
from torch.utils.data import DataLoader
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
        wandb.init(project="test", entity="dchamerslag")
        wandb.config.update(args)

    if args.device=="cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(0)

    # Rescale images (they are all variable dimensions)
    transform = transforms.Compose([
        Rescale((args.rescale_w, args.rescale_h)),
        ToTensor()
    ])
    dataset = AIROGSLiteDataset(transform)
   
    model, optimizer, scheduler = get_model(args.model)
    loss_fn = nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    model = model.to(device)

    training_config = {}
    training_config['dataloader'] = dataloader
    training_config['model'] = model
    training_config['optimizer'] = optimizer
    training_config['scheduler'] = scheduler
    training_config['dataset_size'] = len(dataset)
    training_config['num_epochs'] = args.num_epochs
    training_config['loss_fn'] = loss_fn
    training_config['device'] = device
    training_config['logging'] = args.logging

    trainer = Trainer(training_config)
    model = trainer.train()

    torch.save(model.state_dict(), "model.bin")

if __name__ == "__main__":  
    args = parse_args()
    main(args)