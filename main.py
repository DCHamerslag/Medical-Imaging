
from utils.dataset import AIROGSLiteDataset, Rescale, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from models import get_pretrained_resnet50
from utils.parser import parse_args
from trainer import Trainer
import wandb
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

def main(args):


    transform = transforms.Compose([
        Rescale((200, 200)),
        ToTensor()
    ])
    training_config = {}
    dataset = AIROGSLiteDataset(transform)
   

    model = get_pretrained_resnet50().to('cpu')

    plist = [
            {'params': model.layer4.parameters(), 'lr': 1e-5},
            {'params': model.last_linear.parameters(), 'lr': 5e-3}
            ]

    training_config['dataloader'] = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    training_config['model'] = model
    training_config['optimizer'] = optim.Adam(plist, lr=0.001)
    training_config['scheduler'] = lr_scheduler.StepLR(training_config['optimizer'], step_size=10, gamma=0.1)
    training_config['dataset_size'] = len(dataset)
    training_config['num_epochs'] = 20
    training_config['loss_fn'] = nn.BCEWithLogitsLoss()
    training_config['device'] = 'cpu'
    training_config['logging'] = args.logging

    if args.logging:
        wandb.init(project="test", entity="dchamerslag", config=training_config)
        
    trainer = Trainer(training_config)
    model = trainer.train()

    torch.save(model.state_dict(), "model.bin")

if __name__ == "__main__":  
    args = parse_args()
    main(args)