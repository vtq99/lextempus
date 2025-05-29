import re
import os
import pickle
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PREPROCESSED_FILE = 'uk_cases.pkl'
PATH = './data'
RAW_DATA_FILE = 'uk_cases.jsonl'
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
    import requests
    from bs4 import BeautifulSoup
    from dateutil.parser import parse

    df = pd.concat((pd.read_json('./data/uk_courts_cases/train.jsonl', lines=True),
                    pd.read_json('./data/uk_courts_cases/validation.jsonl', lines=True),
                    pd.read_json('./data/uk_courts_cases/test.jsonl', lines=True)))
    df['date'] = [parse('01/01/1700').date()] * len(df.index)
    valid_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                    "October", "November", "December"]
    pattern = r'\d{1,2}(?:st|nd|rd|th)?\s*(' + '|'.join(valid_months) + '),?\s*(\d{4})'
    for idx in tqdm(range(len(df)), desc='Getting data'):
        id = df.iloc[idx, 0]
        url = f'https://www.bailii.org{id}'
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        inputs = soup.find_all("small")
        try:
            matches = list(re.finditer(pattern, str(inputs[0]), re.IGNORECASE))
            df.iloc[idx, 3] = parse(matches[0].group(0)).date()
        except:
            pass
        if df.iloc[idx, 3] > parse('01/01/2024').date() or df.iloc[idx, 3] < parse('01/01/1701').date():
            input_tags = soup.find_all("date")
            for input_tag in input_tags:
                try:
                    df.iloc[idx, 3] = parse(input_tag.string).date()
                    break
                except:
                    try:
                        matches = list(re.finditer(pattern, str(input_tag), re.IGNORECASE))
                        df.iloc[idx, 3] = parse(matches[0].group(0)).date()
                        break
                    except:
                        try:
                            matches = list(re.finditer(r'\d{2}/\d{2}/\d{2,4}', str(input_tag), re.IGNORECASE))
                            df.iloc[idx, 3] = parse(matches[0].group(0)).date()
                            break
                        except:
                            if '/ew/cases/EWHC/Admin/2013/2281.html' in id:
                                df.iloc[idx, 3] = parse('28/06/2013').date()
                            elif '/ew/cases/EWHC/QB/2008/285.html' in id:
                                df.iloc[idx, 3] = parse('31/12/2008').date()
                            elif '/ew/cases/EWCA/Crim/2006/10.html' in id:
                                df.iloc[idx, 3] = parse('31/12/2006').date()
                            elif '/ew/cases/EWCA/Civ/2002/2036.html' in id:
                                df.iloc[idx, 3] = parse('05/12/2002').date()
                            elif '/ew/cases/EWHC/Ch/2013/2878.html' in id:
                                df.iloc[idx, 3] = parse('22/03/2008').date()
                            else:
                                try:
                                    matches = list(re.finditer(pattern, df.iloc[idx, 2], re.IGNORECASE))
                                    latest = parse(matches[0].group(0))
                                    for match in matches:
                                        now = parse(match.group(0))
                                        if now <= parse(f'31/12/{df.iloc[idx, 1]}'):
                                            if now > latest: latest = now
                                    df.iloc[idx, 3] = latest.date()
                                    break
                                except:
                                    df.iloc[idx, 3] = parse('01/01/1700').date()
                                    print("Can't find ", id, str(input_tag))

        if idx % 1000 == 0:
            df.to_json(os.path.join(PATH, RAW_DATA_FILE) + '_tmp', orient='records', lines=True)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)
    df.to_json(os.path.join(PATH, RAW_DATA_FILE), orient='records', lines=True)


raw_data_path = os.path.join(PATH, RAW_DATA_FILE)
if not os.path.isfile(raw_data_path):
    download_eu()
print('Hehehe')
base_df = pd.read_json(raw_data_path, lines=True)
# Load data frame from json file, group by year
# base_df['year'] = pd.DatetimeIndex(base_df['date']).year
base_df['date'] = pd.DatetimeIndex(base_df['date']).date
base_df = base_df.sort_values(by=['date'])

dataset = []

base_df['text'] = base_df['text'].apply(clean_sentences)
for idx in tqdm(base_df.index, desc='Generating data'):
    train_samples = divide_chunks(base_df['text'][idx], STRIDE, eval=False)
    test_samples = divide_chunks(base_df['text'][idx], STRIDE, eval=True)
    dataset.append({'train': train_samples, 'test': test_samples})

preprocessed_data_path = os.path.join(PATH, PREPROCESSED_FILE)
pickle.dump(dataset, open(preprocessed_data_path, 'wb'))
