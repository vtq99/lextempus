import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from river import cluster, metrics
from river.stream import iter_array
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# batch_size = 64
# samples = pd.read_pickle('./data/echr.pkl')
#
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# sample = [i['test'] for i in samples]
#
# states = []
# avg_states = []
# for s in (tqdm(sample, desc=f"Training")):
#     s = [i[-1] for i in s]
#     embeddings = model.encode(s)
#     avg_states.append(np.mean(embeddings, axis=0))
#     states.append(embeddings)
#
# ppl = {'state': states, 'avg_state': avg_states}
# pickle.dump(ppl, open(f'./data/echr_sentence.pkl', 'wb'))

old = 0
samples = pd.read_pickle('./data/echr_sentence.pkl')


stream = iter_array(X=samples['state'])
avg_stream = iter_array(X=samples['avg_state'])
dbstream = cluster.DBSTREAM(clustering_threshold=0.5,
                              fading_factor=0.01,
                              cleanup_interval=2,
                              intersection_factor=0.3,
                              minimum_weight=1.0)
dbstream_avg = cluster.DBSTREAM(clustering_threshold=0.5,
                              fading_factor=0.01,
                              cleanup_interval=2,
                              intersection_factor=0.3,
                              minimum_weight=1.0)
preds, avg_preds = [], []
distances, avg_distances = [], []
centers, avg_centers = [], []
for i, ((x, y) , (xa, ya)) in tqdm(enumerate(zip(stream, avg_stream)), desc=f"Training"):
    if i > 0:
        avg_preds.append(dbstream_avg.predict_one(xa))
        d = []
        for center in avg_centers[-1].values():
            center = np.array(list(center.values()))
            sample = np.array(list(xa.values()))
            d.append(np.linalg.norm(sample - center))
        avg_distances.append(d)

        d = []
        pred = []
        stream_i = iter_array(np.array(list(x.values())))
        for (xi, yi) in stream_i:
            pred.append(dbstream_avg.predict_one(xi))
            di = []
            for center in centers[-1].values():
                center = np.array(list(center.values()))
                sample = np.array(list(xi.values()))
                di.append(np.linalg.norm(sample - center))
            d.append(di)
        preds.append(pred)
        distances.append(d)

    dbstream_avg.learn_one(xa)
    avg_centers.append(dbstream_avg.centers)
    stream_i = iter_array(np.array(list(x.values())))
    for (xi, yi) in stream_i:
        dbstream.learn_one(xi)
    centers.append(dbstream.centers)

    n_clusters = dbstream_avg.n_clusters
    if i % 1000 == 0 or n_clusters != old:
        print(' Number of clusters:', n_clusters, dbstream.n_clusters)
        old = n_clusters

ppl = {'distance': distances, 'avg_distance': avg_distances, 'pred': preds, 'avg_pred': avg_preds}
pickle.dump(ppl, open(f'./data/echr_cluster.pkl', 'wb'))

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    return np.pad(a, ((0, y_pad), (0, x_pad)), mode='constant')
# a = np.concatenate([to_shape(np.array(distance), (10,7)) for distance in distances])
