import re
import os
import pickle
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PREPROCESSED_FILE = 'eu_cases.pkl'
PATH = './data'
RAW_DATA_FILE = 'eu_cases.jsonl'
GROUP = 2
STRIDE = 390

def remote_between_parenthesis(input_text: str):
    result = re.sub(r'\([^)]*\)', '', input_text)
    return result


def clean_sentences(sentences):
    return remote_between_parenthesis(sentences)


def divide_chunks(law, stride=780, eval=False):
    law = law.split('\n')

    result = []
    cur_len = 0
    max_len = stride

    if eval:
        nxt = []
        for idx, p in enumerate(law):
            if len(p.split()) < 10:
                nxt.append(p)
                continue
            if len(nxt) != 0:
                p = ' '.join(nxt + [p])
            chunk = []
            cur_len = 0
            max_len = stride

            start_idx = idx
            if len(nxt): start_idx = idx - len(nxt)
            nxt = []
            for paragraph in reversed(law[:start_idx]):
                words_no = len(paragraph.split())
                # Add the sample as a list if it is longer than MAX_SEQ_LEN
                if words_no >= max_len and len(chunk) == 0:
                    chunk.append(' '.join(paragraph.split()[-max_len:]))
                    break
                if cur_len + words_no >= max_len:
                    # The next paragraph does not fit
                    chunk = [paragraph] + chunk
                    break
                else:
                    # Add the sample to sequence, continue and try to add more
                    chunk = [paragraph] + chunk
                    cur_len += words_no
                    continue
            if len(chunk) != 0:
                result.append(chunk + [p])
            cur_len = 0
        return result

    cur_len = 0
    max_len = stride * 2

    for idx, p in enumerate(law):
        words_no = len(p.split())
        # Add the sample as a list if it is longer than MAX_SEQ_LEN
        if idx == 0:
            start = 0
        elif (words_no >= stride) or (cur_len + words_no >= stride):
            start = idx - 1
            pass
        else:
            cur_len += words_no
            continue

        chunk = []
        cur_len = 0
        for paragraph in law[start:]:
            words_no = len(paragraph.split())
            # Add the sample as a list if it is longer than MAX_SEQ_LEN
            if words_no >= max_len:
                if len(chunk) > 0:
                    result.append(chunk)
                result.append([paragraph])
                chunk = []
                cur_len = 0
                break
            if cur_len + words_no >= max_len:
                # The next paragraph does not fit in = start for next sequence
                result.append(chunk)
                chunk = []
                cur_len = 0
                break
            else:
                # Add the sample to sequence, continue and try to add more
                chunk.append(paragraph)
                cur_len += words_no
                continue
        if len(chunk) > 0:
            result.append(chunk)
        cur_len = 0
    return result

def download_eu():
    import re
    import requests
    from tqdm import tqdm
    from bs4 import BeautifulSoup
    from datetime import datetime as dt

    # df = pd.concat((pd.read_json('./data/eurlex/train.jsonl', lines=True),
    #                 pd.read_json('./data/eurlex/validation.jsonl', lines=True),
    #                 pd.read_json('./data/eurlex/test.jsonl', lines=True)))
    # df = df[df['descriptor'].isin(['D', 'FJ'])]
    # df['date'] = [dt.strptime('01/01/1700', "%d/%m/%Y").date()]* len(df.index)
    # df = pd.read_json('./data/eu_cases.jsonl_tmp', lines=True)
    df = pd.read_json('./data/eu_cases.jsonl', lines=True)

    for idx in tqdm(range(2657), desc='Getting data'):
        id = df.iloc[idx, 0]
        url = f'https://eur-lex.europa.eu/search.html?scope=EURLEX&text={id}&lang=en&type=quick&qid=1713864980578'
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        input_tags = soup.find_all("dt")
        for i in range(len(input_tags)):
            if 'Date of document' in input_tags[i].string:
                break
        input_tags = soup.find_all("dd")
        try:
            matches = list(re.finditer(r'\d{2}/\d{2}/\d{4}', input_tags[i].string, re.IGNORECASE))
            df.iloc[idx, 5] = dt.strptime(matches[0].group(0), "%d/%m/%Y").date()
        except:
            if '32008D0430' in id:
                df.iloc[idx, 5] = dt.strptime('26/05/2008', "%d/%m/%Y").date()
            elif '32018D1761' in id:
                df.iloc[idx, 5] = dt.strptime('22/11/2018', "%d/%m/%Y").date()
            elif '32017D0620' in id:
                df.iloc[idx, 5] = dt.strptime('01/04/2017', "%d/%m/%Y").date()
            else:
                pass

        if idx % 1000 == 0 and idx > 0:
            df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
            df.to_json(os.path.join(PATH, RAW_DATA_FILE) + '_tmp', orient='records', lines=True)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)
    df.to_json(os.path.join(PATH, RAW_DATA_FILE), orient='records', lines=True)


raw_data_path = os.path.join(PATH, RAW_DATA_FILE)
if not os.path.isfile(raw_data_path):
    download_eu()
base_df = pd.read_json(raw_data_path, lines=True)
# Load data frame from json file, group by year
# base_df['year'] = pd.DatetimeIndex(base_df['date']).year
base_df = base_df.sort_values(by=['date'])

dataset = []

base_df['text'] = base_df['text'].apply(clean_sentences)
for idx in base_df.index:
    train_samples = divide_chunks(base_df['text'][idx], STRIDE, eval=False)
    test_samples = divide_chunks(base_df['text'][idx], STRIDE, eval=True)
    dataset.append({'train': train_samples, 'test': test_samples})

preprocessed_data_path = os.path.join(PATH, PREPROCESSED_FILE)
pickle.dump(dataset, open(preprocessed_data_path, 'wb'))
