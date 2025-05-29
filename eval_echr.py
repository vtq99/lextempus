import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import torch
from tqdm import tqdm

paths = ['./data/echr_gen_zero.pkl', './data/echr_gen_base.pkl', './data/echr_gen_er.pkl',
         './data/echr_gen_adapter.pkl', './data/echr_gen_paramavg.pkl']
file_names = ['./data/echr_scores_zero.pkl', './data/echr_scores_base.pkl',
              './data/echr_scores_er.pkl', './data/echr_scores_adapter.pkl',
              './data/echr_scores_paramavg.pkl']
for path, file_name in zip(paths, file_names):
    samples = pd.read_pickle(path)
    s = pd.read_pickle('./data/echr_test.pkl')
    # file_name = './data/echr_scores_zero.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    all_scores = {}

    # all_scores = pd.read_pickle(file_name)
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
    rouge_scores = {'r1': r1s, 'r2':r2s, 'rl': rls}

    all_scores['rouge'] = rouge_scores
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

    # all_scores = pd.read_pickle(file_name)

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

for path, file_name in zip(paths, file_names):
    samples = pd.read_pickle(path)
    s = pd.read_pickle('./data/echr_test.pkl')
    all_scores = pd.read_pickle(file_name)

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
