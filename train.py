import logging
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from IFmodule import InformationFusionBlock
from get_explanations.vd_oai_model_interface import MODEL
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), acc.item(), tp.item(), tn.item(), fp.item(), fn.item()


class Trainer():
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.initial_f1 = 0.0

        self.text_seq_len = args.max_length

        criteria_filepath = f"criteria/{args.task_id}/final_prompt_set_{MODEL}.json"
        try:
            with open(criteria_filepath, 'r', encoding='utf-8') as f:
                final_criteria = json.load(f)
            num_criteria = len(final_criteria)
        except Exception as e:
            print(f"Error loading criteria file: {e}. Defaulting to 0 criteria.")
            num_criteria = 0

        IF = InformationFusionBlock(
            hidden=768,
            dropout=0.2,
            args=args,
            num_layers=num_criteria
        )
        self.optimizer = optim.AdamW(IF.parameters(), lr=args.lr_IF)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        IF = torch.nn.DataParallel(IF, device_ids=[0, 1, 2, 3])
        self.model = IF.to(args.device)

        self.results_data = []
        if args.resume_file:
            assert os.path.exists(args.resume_file), 'checkpoint not found!'
            logger.info('loading model checkpoint from %s..' % args.resume_file)
            checkpoint = torch.load(args.resume_file)
            IF.load_state_dict(checkpoint['state_dict'], strict=False)

    def savemodel(self, k):
        if not os.path.exists(os.path.join(self.args.savepath)):
            os.mkdir(os.path.join(self.args.savepath))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.savepath, f'model.pth'))

    def train_classicication(self, dataset):
        train_loader = DataLoader(dataset[0], batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=32, pin_memory=True)
        test_loader = DataLoader(dataset[1], batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                 num_workers=32, pin_memory=True)
        for epoch in range(self.start_epoch, self.args.epoch + self.start_epoch):

            self.train_cla_epoch(epoch, train_loader)
            f1 = self.eval_epoch(test_loader)

            if f1 > self.initial_f1:
                self.savemodel(epoch)
                self.initial_f1 = f1

        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall', 'acc'])
        save_path = self.args.savepath + '/result_record_val_' + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)

    def train_cla_epoch(self, epoch, train_loader):
        self.model.train()

        loss_num = 0.0
        all_labels = []
        all_preds = []

        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (code_tokens, desc_tokens, label) in enumerate(pbar):
            woc_flag = self.args.exp_cfg == 'woc'
            outputs = self.model(code_tokens, desc_tokens, woc=woc_flag)
            label = label.to(self.args.device)

            loss = self.criterion(outputs, label)

            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()

            loss_num += loss.sum().item()
            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

    def eval_epoch(self, dev_loader):
        self.model.eval()

        all_labels = []
        all_preds = []

        batch_counter = 0

        with torch.no_grad():
            for code_tokens, desc_tokens, label in tqdm(dev_loader, desc="Evaluating"):
                batch_counter += 1
                for key in code_tokens:
                    code_tokens[key] = code_tokens[key].to(self.args.device, non_blocking=True)

                for desc_dict in desc_tokens:
                    for key in desc_dict:
                        desc_dict[key] = desc_dict[key].to(self.args.device, non_blocking=True)

                woc_flag = self.args.exp_cfg == 'woc'
                outputs = self.model(code_tokens, desc_tokens, woc=woc_flag)
                _, predicted = torch.max(outputs.data, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        if self.args.task_id in ["Devign", "Reveal", "SCVD"]:
            possible_labels = [0, 1]
            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, acc, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)

            self.results_data.append([f1, precision, recall, acc])
        else:
            if self.args.task_id == "DefectPre":
                possible_labels = list(range(4))
            elif self.args.task_id == "POJ-104":
                possible_labels = list(range(104))
            elif self.args.task_id == "Authorship":
                possible_labels = list(range(66))
            else:
                logger.error("task_id error")
                assert False
            acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, labels=possible_labels, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, labels=possible_labels, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, labels=possible_labels, average='macro', zero_division=0)
            self.results_data.append([f1, precision, recall, acc])
        return f1