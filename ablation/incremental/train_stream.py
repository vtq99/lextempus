import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader

# import deepspeed

import pandas as pd

batch_size = 4
samples = pd.read_pickle('./data/eu_cases.pkl')
# samples_test = pd.read_pickle('/srv/querel/thesis/samples_test.pkl')

# Load pre-trained GPT-2 model
model_name = 'gpt2-medium' # You can choose a different GPT-2 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


class MyDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]  # Assuming each element in the Series is the text
        text = ' '.join(text)
        tokens = self.tokenizer(text,
                                padding='max_length',
                                truncation=True, max_length=1024,
                                return_tensors='pt')
        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

# all_years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2019, 2020, 2021, 2022]
# train_years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016]
# test_years = [2019, 2020, 2021, 2022]
all_years = [1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012,
             2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
train_years = [1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012]
test_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
# samples_train = []
# for year in train_years:
#     samples_train += samples[year]
# train_dataset = MyDataset(tokenizer, samples_train)
# val_dataset = MyDataset(tokenizer, samples[2018])
#
# # Create DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

from transformers import AdamW, get_linear_schedule_with_warmup

# Set device and DeepSpeed configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
if 'mps' in device.type:
    batch_size = 1

model.to(device)

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


# model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config)

from transformers import AdamW, get_linear_schedule_with_warmup

# optimizer = AdamW(model.parameters(), lr=5e-5)

# from tqdm.notebook import tqdm
from tqdm import tqdm

import math

# Training loop
ppl_dct = {}
for i, year in enumerate(train_years):
    train_dataset = MyDataset(tokenizer, samples[year])
    val_dataset = MyDataset(tokenizer, samples[all_years[i+1]])
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=deepspeed_config["optimizer"]["params"]["lr"],
                                 weight_decay=deepspeed_config["optimizer"]["params"]["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

    ppl_dct[year] = {}
    for epoch in range(1, 2):
        # Training step
        step = 0
        model.train()
        loss_all = []
        ppl_all = []
        for batch in tqdm(train_dataloader, desc=f"{year} Epoch {epoch}"):
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
                average_loss = np.mean(loss_all)
                perplexity = math.exp(average_loss)
                ppl_all.append(perplexity)
        average_loss = np.mean(loss_all)
        perplexity = math.exp(average_loss)
        ppl_all.append(perplexity)
        print(f'Train loss: {average_loss}, Train ppl: {perplexity}')

        # Validation step
        model.eval()
        loss_all = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Val {epoch}"):
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                target_ids = inputs.clone()
                target_ids[~masks.bool()] = -100
                outputs = model(inputs, labels=target_ids)
                loss = outputs.loss
                loss_all.append(loss.item())
            average_loss = np.mean(loss_all)
            perplexity = math.exp(average_loss)
            ppl_all.append(perplexity)
            print(f'Val loss: {average_loss}, Val ppl: {perplexity}')
        print(f"Epoch {epoch}/{5}: Perplexity: {perplexity}")
        ppl_dct[year][epoch] = ppl_all

        # model.save_pretrained(f'./checkpoints/thesis_deepspeed2_{year}.pt')

print(ppl_dct)

loss_all_year = []
for year in test_years:
    test_dataset = MyDataset(tokenizer, samples[year])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # model_name = f'./checkpoints/thesis_deepspeed2_{train_years[-1]}.pt'  # You can choose a different GPT-2 variant
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.to(device)
    model.eval()
    loss_all = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Test {year}"):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            target_ids = inputs.clone()
            target_ids[~masks.bool()] = -100
            outputs = model(inputs, labels=target_ids)
            loss = outputs.loss
            loss_all.append(loss.item())
        average_loss = np.mean(loss_all)
        perplexity = math.exp(average_loss)
        print(f'{year} loss: {average_loss}, {year} ppl: {perplexity}')
        loss_all_year.append(average_loss)

average_loss = np.mean(loss_all_year)
perplexity = math.exp(average_loss)
print(f'All year loss: {average_loss}, All year ppl: {perplexity}')