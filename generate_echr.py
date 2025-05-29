import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader

# import deepspeed
import numpy as np
import pandas as pd

samples = pd.read_pickle('./data/echr_test.pkl')
batch_size = 64

# Load pre-trained GPT-2 model
# model_name = 'gpt2-medium' # You can choose a different GPT-2 variant
# model = AutoModelForCausalLM.from_pretrained(model_name)

# model_name = './checkpoints/echr_online_adapterfreeze5.pt'
model_name = './checkpoints/echr_online_paramavg_zscore_drift.pt'
model = AutoModelForCausalLM.from_pretrained(model_name)
from adapters import AutoAdapterModel
adapter_model = AutoAdapterModel.from_pretrained(model_name)
import adapters
adapters.init(model)
model.active_adapters = "bottleneck_adapter"

params_src = adapter_model.named_parameters()
params_dest = model.named_parameters()
dict_dest = dict(params_dest)
for name, param in params_src:
    if name in dict_dest:
        dict_dest[name].data.copy_(param.data)

tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = 'left'
tokenizer.padding_side = 'left'
# Set device and DeepSpeed configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
if 'mps' in device.type or 'cpu' in device.type:
    batch_size = 3

# import adapters
import adapters.composition as ac
# adapters.init(model) # prepare model for use with adapters
# model.load_adapter(f'./checkpoints/paramavg_14000.pt', load_as=['bottleneck'], with_head=False)
# model.active_adapters =['bottleneck']

class TestDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        return f' {tokenizer.eos_token} '.join(text[:-1]) + f' {tokenizer.eos_token}'
        # text = self.data[idx]  # Assuming each element in the Series is the text
        # tokens = self.tokenizer(f' {tokenizer.eos_token} '.join(text[:-1]) + f' {tokenizer.eos_token}',
        #                         padding='max_length',
        #                         truncation=True, max_length=512,
        #                         return_tensors='pt')
        # return tokens
        # return {'input_ids': tokens['input_ids'],
        #         'attention_mask': tokens['attention_mask']}

train_dataloaders = []
test_dataloaders = []
num_training_steps = 0
for sample in samples:
    test_dataset = TestDataset(tokenizer, sample['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataloaders.append(test_dataloader)

# Prepare lists to store prompts, targets, and generated texts
generated_text_list, text_list, context_list = [], [], []
# import pandas as pd
# old = pd.read_pickle('./data/echr_gen_zero_old.pkl')
# generated_text_list, text_list, context_list = old['pred'], old['pred'], old['pred']

from tqdm import tqdm
model.to(device)
model.eval()
# weights = pd.read_pickle('./data/echr_test_weights.pkl')
# weight = weights['weight']
# adapter_names = ['0', '3300', '4700', '6200', '7600', '8900', '11400']
# for adapter_name in adapter_names:
#     model.active_adapters = adapter_name
for idx in (tqdm(range(len(samples)), desc=f"Training")):
    generated_text, text, context = [], [], []
    # w = weight[idx].tolist()
    # model.average_adapter('avg', adapter_names, weights=w)
    for i, batch in enumerate(test_dataloaders[idx]):
        # inputs = torch.reshape(batch['input_ids'], (batch['input_ids'].shape[0], 512)).to(device)
        # masks = torch.reshape(batch['attention_mask'], (batch['input_ids'].shape[0], 512)).to(device)
        tokens = tokenizer(batch,
                                padding='max_length',
                                truncation=True, max_length=512,
                                return_tensors='pt')
        tokens = tokens.to(device)
        with torch.no_grad():
            # outputs = model.generate(inputs, attention_mask=masks, max_new_tokens=512, num_return_sequences=1,
            #                          do_sample=True, temperature=0.7, top_k=50, repetition_penalty=2.0)
            outputs = model.generate(**tokens,  max_new_tokens=512, num_return_sequences=1,
                                     do_sample=True, temperature=0.7, top_k=50, repetition_penalty=2.0)
        generated_text += tokenizer.batch_decode(outputs[:, 512:], skip_special_tokens=False)
        context += tokenizer.batch_decode(outputs[:, :512], skip_special_tokens=False)
        text += [t[-1] for t in samples[idx]['test'][i * len(batch):i + len(batch)]]
    generated_text_list.append(generated_text)
    text_list.append(text)
    context_list.append(context)
    # model.delete_adapter('avg')

    # if idx % 50 == 0 and idx>0:
    #     import pickle
    #     df = {'context': context_list, 'pred': generated_text_list, 'true': text_list}
    #     pickle.dump(df, open(f'./data/echr_gen_adapter.pkl', 'wb'))

import pickle
df = {'context': context_list , 'pred': generated_text_list, 'true': text_list}
pickle.dump(df, open(f'./data/echr_gen_adapter.pkl', 'wb'))

del model
del tokens
del outputs

import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import torch
from tqdm import tqdm

path = './data/echr_gen_adapter.pkl'
samples = pd.read_pickle(path)
s = pd.read_pickle('./data/echr_test.pkl')
file_name = './data/echr_scores_adapter.pkl'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# ROUGE SCORE
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1s, r2s, rls = [], [], []
for i in tqdm(range(len(samples['pred'])), desc=f"{path} ROUGE"):
    r1, r2, rl = [], [], []
    for j in range(len(samples['pred'][i])):
        scores = scorer.score(s[i]['test'][j][-1].lower(), samples['pred'][i][j].lower())
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
    r1s.append(sum(r1) / len(r1))
    r2s.append(sum(r2) / len(r2))
    rls.append(sum(rl) / len(rl))
rouge_scores = {'r1': r1s, 'r2': r2s, 'rl': rls}

all_scores = {'rouge': rouge_scores}
pickle.dump(all_scores, open(file_name, 'wb'))

# BERT SCORE
from bert_score import score as bert_score

precisions, recalls, f1s = [], [], []
for i in tqdm(range(len(samples['pred'])), desc=f"{path} BERT"):
    P, R, F1 = bert_score(samples['pred'][i], [t[-1] for t in s[i]['test']], lang='en', verbose=False, device=device)
    precisions.append(float(P.mean()))
    recalls.append(float(R.mean()))
    f1s.append(float(F1.mean()))
bert_scores = {'precision': precisions, 'recall': recalls, 'f1': f1s}

all_scores['bert'] = bert_scores
pickle.dump(all_scores, open(file_name, 'wb'))

# ALIGN SCORE
from alignscore import AlignScore
scorer = AlignScore(model='roberta-base', batch_size=64, device=device, ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt', evaluation_mode='nli')
scores = []
for i in tqdm(range(len(samples['pred'])), desc=f"{path} ALIGN"):
    score = scorer.score(contexts=[t[-1] for t in s[i]['test']], claims=samples['pred'][i])
    scores.append(sum(score)/len(score))
align_scores = {'score': scores}

all_scores['align'] = align_scores
pickle.dump(all_scores, open(file_name, 'wb'))

# UNIEVAL
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
coherences, consistencys, fluencys, relevances, overalls = [], [], [], [], []
task = 'summarization'
evaluator = get_evaluator(task, device=device)

for i in tqdm(range(len(samples['pred'])), desc=f"{path} UNIEVAL"):
    coherence, consistency, fluency, relevance, overall = 0, 0, 0, 0, 0
    src_list = samples['context'][i]
    ref_list = [t[-1] for t in s[i]['test']]
    output_list = samples['pred'][i]

    data = convert_to_json(output_list=output_list,
                           src_list=src_list, ref_list=ref_list)
    eval_scores = evaluator.evaluate(data, print_result=False)
    for i in range(len(eval_scores)):
        coherence += eval_scores[i]['coherence']
        consistency += eval_scores[i]['consistency']
        fluency += eval_scores[i]['fluency']
        relevance += eval_scores[i]['relevance']
        overall += eval_scores[i]['overall']
    coherences.append(round(coherence / len(eval_scores), 6))
    consistencys.append(round(consistency / len(eval_scores), 6))
    fluencys.append(round(fluency / len(eval_scores), 6))
    relevances.append(round(relevance / len(eval_scores), 6))
    overalls.append(round(overall / len(eval_scores), 6))
unieval_scores = {'coherence': coherences, 'consistency': consistencys, 'fluency': fluencys, 'relevance': relevances, 'overall': overalls}

all_scores['unieval'] = unieval_scores
pickle.dump(all_scores, open(file_name, 'wb'))
