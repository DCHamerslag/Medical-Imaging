from typing import Any, Dict
import torch
import wandb
from tqdm import tqdm

class Trainer():
    def __init__(self, config: Dict):
        self.model = config['model']
        self.dataloader = config['dataloader']
        self.test_dataloader = config['test_dataloader']
        self.dataset_size = config['dataset_size']
        self.optimizer = config['optimizer']
        self.scheduler = config['scheduler']
        self.num_epochs = config['num_epochs']
        self.loss_fn = config['loss_fn']
        self.device = config['device']
        self.logging = config['logging']
        self.batch_size = config['batch_size']

    def train(self):

        train_pb = tqdm(total=int(self.dataset_size / self.batch_size), desc="Training progress")
        for epoch in tqdm(range(self.num_epochs - 1), "Epoch"):
            train_pb.reset()

            self.model.train()

            running_loss = 0.0
            
            correct = {}
            for batch_index, batch in enumerate(self.dataloader):
                train_pb.update()
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
                #print(outputs)
                #print(labels)
                #print(torch.argmax(outputs, 1))
                #print(torch.argmax(labels, dim=1))
                preds = torch.argmax(outputs, dim=1)
                truth = torch.argmax(labels, dim=1)

                correct = preds[preds==truth]
                incorrect = preds[preds!=truth]
                true_pos = torch.count_nonzero(correct)
                true_neg = correct.shape[0] - true_pos
                acc = (true_pos + true_neg) / self.batch_size
                #print(f"True pos {true_pos}, true neg: {true_neg}")
                running_loss += loss.item() * inputs.size(0)
                train_pb.set_postfix(loss=loss.item(), acc=acc.item())
                if self.logging: 
                    wandb.log({"Loss" : loss,
                                "TP" : true_pos,
                                "TN" : true_neg,
                                "Acc" : acc
                    })
            epoch_loss = running_loss / self.dataset_size
            #print('Epoch Loss: {:.4f}'.format(epoch_loss))
            if self.logging: wandb.log({"Epoch loss" : epoch_loss})

        return self.model