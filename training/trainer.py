
from typing import Any, Dict
import torch
import wandb
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import sys

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
        self.model_name = config['model_name']

    def train(self):

        train_pb = tqdm(total=int(self.dataset_size / self.batch_size), desc="Training progress")
        max_mean_scores = 0
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

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                TP, TN, FP, FN = update_metrics(outputs, labels, TP, TN, FP, FN)
                acc = (TP + TN) / (self.batch_size * (batch_index + 1))
                running_loss += loss.item() * inputs.size(0)
                train_pb.set_postfix(loss=loss.item(), acc=acc.item())
                if self.logging:
                    wandb.log({"Loss" : loss,
                                "TP (%)" : TP / (self.batch_size * (batch_index + 1)),
                                "TN (%)" : TN / (self.batch_size * (batch_index + 1)),
                                "FN (%)" : FN / (self.batch_size * (batch_index + 1)),
                                "FP (%)" : FP / (self.batch_size * (batch_index + 1)),
                                "Acc" : acc
                    })
            self.scheduler.step()
            epoch_loss = running_loss / self.dataset_size
            epoch_accuracy = acc

            # validation set
            with torch.no_grad():
                self.model.eval()
                TP = TN = FP = FN = 0
                all_outputs = []
                all_labels = []
                for test_batch_index, test_batch in enumerate(self.test_dataloader):
                    inputs = test_batch["image"]
                    labels = test_batch["label"]
                    inputs = inputs.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device, dtype=torch.float)
                    outputs = self.model(inputs)
                    to_probabilities = nn.Softmax(dim=1)
                    probs = to_probabilities(outputs)
                    all_outputs.append(probs)
                    all_labels.append(labels)
                    TP, TN, FP, FN = update_metrics(outputs, labels, TP, TN, FP, FN)
                all_outputs = torch.cat(all_outputs)
                all_labels = torch.cat(all_labels)
                if self.device != "cpu":
                    all_outputs = all_outputs.cpu()
                    all_labels = all_labels.cpu()
                partial_auroc = metrics.roc_auc_score(all_labels, all_outputs, max_fpr=(1 - 0.9))
                sens_at_95spec = screening_sens_at_spec(all_outputs, all_labels, 0.95)
                val_acc = (TP + TN) / (TP + TN + FP + FN)

            #print('Epoch Loss: {:.4f}'.format(epoch_loss))
            if self.logging:
                wandb.log({"Epoch loss" : epoch_loss,
                           "Epoch accuracy" : epoch_accuracy,
                           "Validation TP" : TP,
                           "Validation TN" : TN,
                           "Validation FP" : FP,
                           "Validation FN" : FN,
                           "Validation Accuracy" : val_acc,
                           "Partial AUROC 90-100% spec" : partial_auroc,
                           "Sensitivity @ 95% specificity" : sens_at_95spec})
            mean_scores = (partial_auroc+sens_at_95spec)/2
            if mean_scores > max_mean_scores:
                max_mean_scores = mean_scores
                if (self.model_name == ''):
                    model_name = 'model.bin'
                else:
                    model_name = self.model_name
                torch.save(self.model.state_dict(), model_name)

        return self.model

def update_metrics(outputs: torch.TensorType, labels: torch.TensorType,
        TP: int, TN: int, FP: int, FN: int):

    preds = torch.argmax(outputs, dim=1)
    truth = torch.argmax(labels, dim=1)

    num_pos = torch.sum(preds)
    num_neg = preds.shape[0] - num_pos
    correct = preds[preds==truth]
    incorrect = preds[preds!=truth]
    TP_b = torch.count_nonzero(correct)
    TN_b = correct.shape[0] - TP_b
    FP += num_pos - TP_b
    FN += num_neg - TN_b
    TP += TP_b
    TN += TN_b
    return TP, TN, FP, FN

def update_metrics_fixed(outputs: torch.TensorType, labels: torch.TensorType,
        TP: int, TN: int, FP: int, FN: int):

    preds = torch.argmax(outputs.detach(), dim=1)
    truth = torch.argmax(labels.detach(), dim=1)
    for i, _ in enumerate(preds):
        
        if preds[i] == 1 and truth[i] == 1:
            TP += 1
        elif preds[i] == 1 and truth[i] == 0:
            FP += 1
        elif preds[i] == 0 and truth[i] == 0:
            TN += 1
        elif preds[i] == 0 and truth[i] == 1:
            FN += 1

    return TP, TN, FP, FN

def screening_sens_at_spec(y_pred, y_true, at_spec, eps=sys.float_info.epsilon):
    fpr, tpr, threshes = metrics.roc_curve(y_true[:,1], y_pred[:,1], drop_intermediate=False)
    spec = 1 - fpr

    operating_points_with_good_spec = spec >= (at_spec - eps)
    max_tpr = tpr[operating_points_with_good_spec][-1]

    operating_point = np.argwhere(operating_points_with_good_spec).squeeze()[-1]
    operating_tpr = tpr[operating_point]

    assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
    assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
        f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

    return max_tpr