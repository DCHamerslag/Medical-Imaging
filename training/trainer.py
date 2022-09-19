from typing import Dict
import torch
import wandb

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


    def train(self) -> any: # idk model datatype

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            self.model.train()

            running_loss = 0.0
            for batch_index, batch in enumerate(self.dataloader):
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
                print('Current loss: {:.4f}'.format(loss))
                if self.logging: wandb.log({"Loss" : loss})
            epoch_loss = running_loss / self.dataset_size
            print('Loss: {:.4f}'.format(epoch_loss))

        return self.model