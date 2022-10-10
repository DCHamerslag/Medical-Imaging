
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
            
            TP = TN = FN = FP = 0
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
                TP, TN, FP, FN = update_metrics(outputs, labels, TP, TN, FP, FN)
                acc = (TP + TN) / (self.batch_size * (batch_index + 1))
                running_loss += loss.item() * inputs.size(0)
                train_pb.set_postfix(loss=loss.item(), acc=acc.item())
                if self.logging: 
                    wandb.log({"Loss" : loss,
                                "TP (%)" : TP / self.batch_size * (batch_index + 1),
                                "TN (%)" : TN / self.batch_size * (batch_index + 1),
                                "FN (%)" : FN / self.batch_size * (batch_index + 1),
                                "FP (%)" : FP / self.batch_size * (batch_index + 1),
                                "Acc" : acc
                    })
            epoch_loss = running_loss / self.dataset_size
            epoch_accuracy = acc
            #print('Epoch Loss: {:.4f}'.format(epoch_loss))
            if self.logging: wandb.log({"Epoch loss" : epoch_loss,
                                        "Epoch accuracy" : epoch_accuracy})

        return self.model

def update_metrics(outputs: torch.TensorType, labels: torch.TensorType,
        TP: int, TN: int, FP: int, FN: int):

    preds = torch.argmax(outputs, dim=1)
    truth = torch.argmax(labels, dim=1)

    num_pos = torch.sum(labels[:,1])
    num_neg = torch.sum(labels[:,0])
    correct = preds[preds==truth]
    incorrect = preds[preds!=truth]
    TP_b = torch.count_nonzero(correct)
    TN_b = correct.shape[0] - TP_b
    FN += num_pos - TP_b
    FP += num_neg - TN_b
    TP += TP_b
    TN += TN_b
    return TP, TN, FP, FN