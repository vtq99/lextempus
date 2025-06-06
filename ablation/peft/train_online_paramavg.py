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
import pickle


batch_size = 6
samples = pd.read_pickle('./data/echr.pkl')
weights = pd.read_pickle('./data/echr_gpt_zscore.pkl')
weight = weights['weight']
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

import adapters
import adapters.composition as ac
adapters.init(model) # prepare model for use with adapters
for name, child in model.named_children():
    if 'lm_head' not in name:
        for param in child.parameters():
            param.requires_grad = False
# adapter_names = ['0', '2000', '4000', '6000', '8000', '10000', '12000', '14000']
# adapter_names = ['0', '4000', '8000', '12000']
adapter_names = ['0', '3900', '6500', '7800', '8900']
# adapter_names = ['0', '3000', '4300', '6000', '7100', '8300', '9800', '11400', '14100']
new_adapters = ['0']
model.add_adapter('0')
model.active_adapters = '0'


print(sum(p.numel() for p in model.parameters()), ' params, trainable: ',
      sum(p.numel() for p in model.parameters() if p.requires_grad))


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

# from RecAdam import RecAdam, anneal_function
# # Prepare for the grouped parameters for RecAdam optimizer.
# no_decay = ["bias", "LayerNorm.weight"]
# # Since the classifier layer is not pretrained, it is not penalized during optimization.
# optimizer_grouped_parameters = [
#     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": deepspeed_config["optimizer"]["params"]["weight_decay"],
#         "anneal_w": 1.0,
#         "pretrain_params": [p_p for p_n, p_p in model.named_parameters() if not any(nd in p_n for nd in no_decay)]}
# ]
# optimizer = RecAdam(optimizer_grouped_parameters, lr=deepspeed_config["optimizer"]["params"]["lr"], eps=1e-8,
#                     anneal_fun='sigmoid', anneal_k=0.5,
#                     anneal_t0=250, pretrain_cof=5000.0)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=deepspeed_config["optimizer"]["params"]["lr"],
                             weight_decay=deepspeed_config["optimizer"]["params"]["weight_decay"]
                             )

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
    # if idx not in [0, 1, 3899, 3900, 3901, 6499, 6500, 6501, 7799, 7800,7801, 8899, 8900, 8901]:
    #     continue

    if idx % 1000 == 0 and idx > 0:
        print("Moving avg test ppl: ", np.nanmean(test_ppl_all[-1000:]))

    if str(idx) in adapter_names[1:]:
        ppl = {'train': ppl_all, 'test': test_ppl_all}
        pickle.dump(ppl, open(f'./checkpoints/echr_online_paramavg_zscore_unfreeze.pkl', 'wb'))
        new_adapters = adapter_names[:adapter_names.index(str(idx)) + 1]
        model.save_adapter(f'./checkpoints/paramavg1_{new_adapters[-2]}.pt', new_adapters[-2])
        # for name, child in model.named_children():
        #     if 'lm_head' not in name:
        #         for param in child.parameters():
        #             param.requires_grad = False
        model.load_adapter(f'./checkpoints/paramavg1_{new_adapters[-2]}.pt', load_as=new_adapters[-1], with_head=False)
        model.active_adapters = new_adapters[-1]
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=deepspeed_config["optimizer"]["params"]["lr"],
                                     weight_decay=deepspeed_config["optimizer"]["params"]["weight_decay"]
                                     )
        model.to(device)
        model.train()
        print(sum(p.numel() for p in model.parameters()), ' params, trainable: ',
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        # if idx == 6500:
        #     batch_size -= 1
        #     train_dataloaders = []
        #     test_dataloaders = []
        #     num_training_steps = 0
        #     for sample in samples:
        #         train_dataset = TrainDataset(tokenizer, sample['train'])
        #         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        #         train_dataloaders.append(train_dataloader)
        #         num_training_steps += len(train_dataloader)
        #
        #         test_dataset = TrainDataset(tokenizer, sample['test'])
        #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        #         test_dataloaders.append(test_dataloader)

    # Training step
    model.eval()
    if idx < int(adapter_names[1]):
        w = [1]
    elif idx < int(adapter_names[2]):
        w = [0.25, 0.75]
    else:
        w = weight[idx].tolist() + [1/(len(new_adapters) -2)]
        # w = weight[idx].tolist() + [1]
    model.average_adapter('avg', new_adapters, weights=w)

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
    model.delete_adapter('avg')

    model.train()
    epochs = 5
    for epoch in range(epochs):
        for batch in train_dataloaders[idx]:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            target_ids = inputs.clone()
            target_ids[~masks.bool()] = -100
            outputs = model(inputs, labels=target_ids)
            loss = outputs.loss
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = np.nanmean(loss_all)
        perplexity = math.exp(average_loss)
        ppl_all.append(perplexity)
        loss_all = []

ppl = {'train': ppl_all, 'test': test_ppl_all}
pickle.dump(ppl, open(f'./checkpoints/echr_online_paramavg_zscore_unfreeze.pkl', 'wb'))
model.save_pretrained(f'./checkpoints/echr_online_paramavg_zscore_unfreeze.pt')

model.save_adapter(f'./checkpoints/paramavg1_{adapter_names[-1]}.pt', adapter_names[-1])
