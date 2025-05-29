import re
import os
import pickle
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PREPROCESSED_FILE = 'echr.pkl'
PATH = './data'
RAW_DATA_FILE = 'echr_cases_total.pkl'
GROUP = 2
STRIDE = 390


def remote_between_parenthesis(input_text: str):
    result = re.sub(r'\([^)]*\)', '', input_text)
    return result


def clean_sentences(sentences):
    return [remote_between_parenthesis(x) for x in sentences]


def divide_chunks(fact, law, stride=780, eval=False):
    result = []
    chunk = []
    cur_len = 0
    max_len = stride

    # Max facts = 512 tokens
    for paragraph in reversed(fact):
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
    # Concat fact and law sections
    law = chunk + law
    if eval:
        law_idx = len(chunk)
        nxt = []
        for idx, p in enumerate(law[law_idx:]):
            if len(p.split()) < 10:
                nxt.append(p)
                continue
            if len(nxt) != 0:
                p = ' '.join(nxt + [p])
            chunk = []
            cur_len = 0
            max_len = stride
            # Max facts = 512 tokens

            start_idx = idx + law_idx
            if len(nxt): start_idx = idx + law_idx - len(nxt)
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


raw_data_path = os.path.join(PATH, RAW_DATA_FILE)
if not os.path.isfile(raw_data_path):
    raise ValueError(f'{raw_data_path} is not in the data directory!')
base_df = pd.read_pickle(raw_data_path)
# Load data frame from json file, group by year
base_df = base_df.sort_values(by=['judgementdate'])

dataset = []

base_df['PCR_REMAINDER_REMAINDER_CLEANED'] = base_df['PCR_REMAINDER_REMAINDER'].apply(clean_sentences)
for idx in base_df.index:
    train_samples = divide_chunks(base_df['PCR_FACTS'][idx], base_df['PCR_REMAINDER_REMAINDER_CLEANED'][idx], STRIDE, eval=False)
    test_samples = divide_chunks(base_df['PCR_FACTS'][idx], base_df['PCR_REMAINDER_REMAINDER_CLEANED'][idx], STRIDE, eval=True)
    dataset.append({'train': train_samples, 'test': test_samples})

preprocessed_data_path = os.path.join(PATH, PREPROCESSED_FILE)
pickle.dump(dataset, open(preprocessed_data_path, 'wb'))
