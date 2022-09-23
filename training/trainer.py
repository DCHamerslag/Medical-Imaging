from typing import Dict
import torch
import wandb
from tqdm import tqdm

class Trainer():
    def __init__(self, config: Dict):
        self.model = config['model']
        self.dataloader = config['dataloader']
        self.dataset_size = config['dataset_size']
        self.optimizer = config['optimizer']
        self.scheduler = config['scheduler']
        self.num_epochs = config['num_epochs']
        self.loss_fn = config['loss_fn']
        self.device = config['device']
        self.logging = config['logging']
        self.batch_size = config['batch_size']

    def train(self) -> any:

        training_progress_bar = tqdm(total=int(self.dataset_size / self.batch_size), desc="Training progress")
        for epoch in tqdm(range(self.num_epochs - 1), "Epoch"):
            training_progress_bar.reset()

            self.model.train()

            running_loss = 0.0
            for batch_index, batch in enumerate(self.dataloader):
                training_progress_bar.update()
                inputs = batch["image"]
                labels = batch["label"]
                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                training_progress_bar.set_postfix(loss=loss.item())
                if self.logging: wandb.log({"Loss" : loss})
            epoch_loss = running_loss / self.dataset_size
            #print('Epoch Loss: {:.4f}'.format(epoch_loss))
            if self.logging: wandb.log({"Epoch loss" : epoch_loss})

        return self.model