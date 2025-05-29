import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# import deepspeed

import pandas as pd

batch_size = 4
samples = pd.read_pickle('./data/echr_long.pkl')
# samples_test = pd.read_pickle('/srv/querel/thesis/samples_test.pkl')

# Load pre-trained GPT-2 model
model_name = 'gpt2-medium' # You can choose a different GPT-2 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained('./checkpoints/thesis_deepspeed2_ep_1.pt')
# Set device and DeepSpeed configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
if 'mps' in device.type:
    batch_size = 2
model.to(device)

class MyDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]  # Assuming each element in the Series is the text
        # text = ' '.join(text)
        # tokens = self.tokenizer(text,
        #                         padding='max_length', truncation=True, max_length=1024,
        #                         # padding=True, pad_to_multiple_of=512,
        #                         return_tensors='pt')
        # return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        tokens = self.tokenizer(' '.join(text[:-1]),
                                padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokens2 = self.tokenizer(text[-1:],
                                padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')
        return {'input_ids': torch.cat((tokens['input_ids'], tokens2['input_ids']), dim=-1),
                'attention_mask': torch.cat((tokens['attention_mask'], tokens2['attention_mask']), dim=-1)}

samples_train = []
train_years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016]
# train_years = [2000, 2002, 2004, 2006]
# train_years = [2012, 2014]
test_years = [2019, 2020, 2021, 2022]
# train_years = [1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012]
# train_years = [1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994]
# train_years = [2008, 2010, 2012]
# test_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
for year in train_years:
    samples_train += samples[year]
train_dataset = MyDataset(tokenizer, samples_train)
val_dataset = MyDataset(tokenizer, samples[2018])
# val_dataset = MyDataset(tokenizer, samples[2014])

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

from transformers import AdamW, get_linear_schedule_with_warmup

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
                             weight_decay=deepspeed_config["optimizer"]["params"]["weight_decay"])

# model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config)

from transformers import AdamW, get_linear_schedule_with_warmup

# optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

# from tqdm.notebook import tqdm
from tqdm import tqdm

import math

# Training loop
for epoch in range(1):
    # Training step
    step = 0
    model.train()
    loss_all = []
    ppl_all = []
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
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
        scheduler.step()

        step += 1
        if step % int(1000 / batch_size) == 0:
            average_loss = np.nanmean(loss_all)
            perplexity = math.exp(average_loss)
            ppl_all.append(perplexity)
    average_loss = np.nanmean(loss_all)
    perplexity = math.exp(average_loss)
    print(ppl_all)
    print(f'Train loss: {average_loss}, Train ppl: {perplexity}')

    # Validation step
    model.eval()
    loss_all = []
    for batch in tqdm(val_dataloader, desc=f"Val {epoch}"):
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        target_ids = inputs.clone()
        target_ids[~masks.bool()] = -100
        target_ids[:, :, :513] = -100
        with torch.no_grad():
            outputs = model(inputs, labels=target_ids)
            loss = outputs.loss
            loss_all.append(loss.item())
    average_loss = np.nanmean(loss_all)
    perplexity = math.exp(average_loss)
    print(f'Val loss: {average_loss}, Val ppl: {perplexity}')

    model.save_pretrained(f'./checkpoints/thesis_deepspeed2_ep_{epoch}.pt')

    loss_all_year = []
    model.eval()
    for year in test_years:
        test_dataset = MyDataset(tokenizer, samples[year])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loss_all = []
        for batch in tqdm(test_dataloader, desc=f"Test {year}"):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            target_ids = inputs.clone()
            target_ids[~masks.bool()] = -100
            target_ids[:, :, :513] = -100
            with torch.no_grad():
                outputs = model(inputs, labels=target_ids)
                loss = outputs.loss
                loss_all.append(loss.item())
        average_loss = np.nanmean(loss_all)
        perplexity = math.exp(average_loss)
        print(f'{year} loss: {average_loss}, {year} ppl: {perplexity}')
        loss_all_year.append(average_loss)

    average_loss = np.nanmean(loss_all_year)
    perplexity = math.exp(average_loss)
    print(f'All year loss: {average_loss}, All year ppl: {perplexity}')
