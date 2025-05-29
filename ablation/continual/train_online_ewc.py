import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# import deepspeed

import pandas as pd
from tqdm import tqdm
import math

batch_size = 4
samples = pd.read_pickle('./data/eu_cases.pkl')
# samples_test = pd.read_pickle('/srv/querel/thesis/samples_test.pkl')

# Load pre-trained GPT-2 model
model_name = 'gpt2-medium' # You can choose a different GPT-2 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# Set device and DeepSpeed configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
if 'mps' in device.type:
    batch_size = 1

ewc_dct = {'lambda': 0.5, 'gamma': 1.0,
           'online': True, 'fisher_n': None, 'emp_FI': False, 'EWC_task_count': 0}
def estimate_fisher(idx):
    """
    After completing training on a task, estimate diagonal of Fisher Information matrix.
    [dataset]:          <DataSet> to be used to estimate FI-matrix
    """
    est_fisher_info = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            est_fisher_info[n] = p.detach().clone().zero_()

    model.eval()
    idx = np.random.randint(idx, size=1)[0]

    ind = 0
    for index, batch in enumerate(train_dataloaders[idx]):
        if ewc_dct['fisher_n'] is not None:
            if index >= ewc_dct['fisher_n']:
                break
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        target_ids = inputs.clone()
        target_ids[~masks.bool()] = -100
        outputs = model(inputs, labels=target_ids)
        negloglikelihood = outputs.loss

        model.zero_grad()
        negloglikelihood.backward()

        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] += p.grad.detach() ** 2
        ind = index

    est_fisher_info = {n: p / (ind + 1) for n, p in est_fisher_info.items()}

    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            model.register_buffer(
                '{}_EWC_prev_task{}'.format(n, "" if ewc_dct['online'] else ewc_dct['EWC_task_count'] + 1),
                p.detach().clone())
            if ewc_dct['online'] and ewc_dct['EWC_task_count'] == 1:
                existing_values = getattr(model, '{}_EWC_estimated_fisher'.format(n))
                est_fisher_info[n] += ewc_dct['gamma'] * existing_values
            model.register_buffer(
                '{}_EWC_estimated_fisher{}'.format(n, "" if ewc_dct['online'] else ewc_dct['EWC_task_count'] + 1),
                est_fisher_info[n])

    ewc_dct['EWC_task_count'] = 1 if ewc_dct['online'] else ewc_dct['EWC_task_count'] + 1

    model.train()


def ewc_loss():
    if ewc_dct['EWC_task_count'] > 0:
        losses = []
        for task in range(1, ewc_dct['EWC_task_count'] + 1):
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    mean = getattr(model, '{}_EWC_prev_task{}'.format(n, "" if ewc_dct['online'] else task))
                    fisher = getattr(model, '{}_EWC_estimated_fisher{}'.format(n, "" if ewc_dct['online'] else task))
                    fisher = ewc_dct['gamma'] * fisher if ewc_dct['online'] else fisher
                    losses.append((fisher * (p - mean) ** 2).sum())
        return (1. / 2) * sum(losses)
    else:
        return torch.tensor(0., device=device)


class TrainDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]  # Assuming each element in the Series is the text
        text = f' {tokenizer.eos_token} '.join(text) + f' {tokenizer.eos_token}'
        tokens = self.tokenizer(text,
                                padding='max_length', truncation=True, max_length=1024,
                                # padding=True, pad_to_multiple_of=512,
                                return_tensors='pt')
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}


class TestDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]  # Assuming each element in the Series is the text

        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        tokens = self.tokenizer(f' {tokenizer.eos_token} '.join(text[:-1]) + f' {tokenizer.eos_token}',
                                padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokens2 = self.tokenizer(text[-1] + f' {tokenizer.eos_token}',
                                padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')
        return {'input_ids': torch.cat((tokens['input_ids'], tokens2['input_ids']), dim=-1),
                'attention_mask': torch.cat((tokens['attention_mask'], tokens2['attention_mask']), dim=-1)}

# Define DeepSpeed configuration as a dictionary
deepspeed_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7,
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 100,
        }
    },
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": 5e8
    }
}

optimizer = torch.optim.Adam(model.parameters(),
                             lr=deepspeed_config["optimizer"]["params"]["lr"],
                             weight_decay=deepspeed_config["optimizer"]["params"]["weight_decay"]
                             )
# model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config)
train_dataloaders = []
test_dataloaders = []
num_training_steps = 0
for sample in samples:
    train_dataset = TrainDataset(tokenizer, sample['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_dataloaders.append(train_dataloader)
    num_training_steps += len(train_dataloader)

    test_dataset = TrainDataset(tokenizer, sample['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataloaders.append(test_dataloader)

# Training loop
step = 0
model.to(device)
model.train()
loss_all, test_loss_all = [], []
ppl_all, test_ppl_all = [], []

for idx in (tqdm(range(len(samples)), desc=f"Training")):
    # Training step
    model.eval()
    for batch in test_dataloaders[idx]:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        target_ids = inputs.clone()
        target_ids[~masks.bool()] = -100
        target_ids[:, :, :513] = -100
        with torch.no_grad():
            outputs = model(inputs, labels=target_ids)
            loss = outputs.loss
            test_loss_all.append(loss.item())
    average_loss = np.nanmean(test_loss_all)
    perplexity = math.exp(average_loss)
    test_ppl_all.append(perplexity)
    test_loss_all = []

    model.train()
    epochs = 1
    for epoch in range(epochs):
        for batch in train_dataloaders[idx]:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            target_ids = inputs.clone()
            target_ids[~masks.bool()] = -100
            outputs = model(inputs, labels=target_ids)
            loss = outputs.loss
            loss = loss + ewc_dct['lambda'] * ewc_loss()
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = np.nanmean(loss_all)
        perplexity = math.exp(average_loss)
        ppl_all.append(perplexity)
        loss_all = []
        if idx > 0:
            estimate_fisher(idx)

    if idx % 10000 == 0 and idx > 0:
        model.save_pretrained(f'./checkpoints/eu_online_ewc.pt')
import pickle
ppl = {'train': ppl_all, 'test': test_ppl_all}
pickle.dump(ppl, open(f'./checkpoints/eu_online_ewc.pkl', 'wb'))
