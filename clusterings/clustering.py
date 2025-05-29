from sklearn import metrics
from sklearn.cluster import DBSCAN
from scipy.special import softmax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# batch_size = 64
# samples = pd.read_pickle('./data/echr.pkl')
#
# # model_name = 'bert-base-uncased' # You can choose a different GPT-2 variant
# model_name = 'nlpaueb/legal-bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.truncation_side = 'left'
# model = AutoModel.from_pretrained(model_name)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# if 'mps' in device.type:
#     batch_size = 32
#
#
# class TrainDataset(Dataset):
#     def __init__(self, tokenizer, data):
#         self.tokenizer = tokenizer
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text = self.data[idx]  # Assuming each element in the Series is the text
#         text = ' '.join(text)
#         tokens = self.tokenizer(text,
#                                 padding='max_length', truncation=True, max_length=512,
#                                 # padding=True, pad_to_multiple_of=512,
#                                 return_tensors='pt')
#         return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
#
#
# dataloaders = []
# num_training_steps = 0
# for sample in samples:
#     dataset = TrainDataset(tokenizer, sample['train'])
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     dataloaders.append(dataloader)
#
# model.to(device)
# model.eval()
#
# states = []
# avg_states = []
# for idx in (tqdm(range(len(samples)), desc=f"Training")):
#     # Training step
#     sub_states = []
#     for batch in dataloaders[idx]:
#         inputs = batch['input_ids'].reshape(batch['input_ids'].shape[0], 512).to(device)
#         masks = batch['attention_mask'].reshape(batch['input_ids'].shape[0], 512).to(device)
#         with torch.no_grad():
#             hidden_state = model(input_ids=inputs, attention_mask=masks)[0]
#             sub_states.append(hidden_state[:, 0].cpu())
#     avg_states.append(np.mean(np.concatenate(sub_states), axis=0))
#     states.append(np.concatenate(sub_states))
#
# ppl = {'state': states, 'avg_state': avg_states}
# pickle.dump(ppl, open(f'./data/echr_legalbert.pkl', 'wb'))

old = 0
all_labels = []
distances = []
samples = pd.read_pickle('./data/echr_gpt.pkl')
# DBSCAN
db = None
for i in range(500):
    if db is not None:
        centroids = []
        for k in range(n_clusters_):
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            class_member_mask = labels == k
            core_states = np.array(states)[class_member_mask & core_samples_mask]
            # core_states = np.array(states)[class_member_mask]
            centroids.append(np.mean(core_states, axis=0))

        db.fit_predict(samples['avg_state'][:(i + 1) * 10])
        preds = db.labels_[-10:]

        dist = []
        for j in range(10):
            d = []
            for k, centroid in enumerate(centroids):
                # if k == preds[j]:
                #     d.append(0)
                # else:
                d.append(np.linalg.norm(states[(i-1)*10 + j] - centroid))
            dist.append(d)
        if n_clusters_ == 0:
            distances.append([[1]] * 10)
        else:
            distances.append(softmax(0 - np.array(dist), axis=1))

    states = samples['avg_state'][:(i + 1) * 10]
    # states = [si for s in samples['state'][:(i + 1) * 10] for si in s]

    db = DBSCAN(eps=5, min_samples=10).fit(states)
    labels = db.labels_
    all_labels.append(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if i%100==0 or n_clusters_ != old:
        print(i, "Estimated number of clusters: %d" % n_clusters_, "Estimated number of noise points: %d" % n_noise_)

    if n_clusters_ == old and i != 157:
        continue
    else:
        old = n_clusters_

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pos = pca.fit_transform(states)
    x_pos, y_pos, z_pos = pos[:, 0], pos[:, 1], pos[:, -1]
    # plt.scatter(x_pos, y_pos)
    # plt.show()
    #
    # ppl = {'x': x_pos, 'y': y_pos, 'label': labels}
    # pickle.dump(ppl, open(f'./data/echr_bert_pca.pkl', 'wb'))

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        # plt.plot(
        #     x_pos[class_member_mask & core_samples_mask],
        #     y_pos[class_member_mask & core_samples_mask],
        #     "o",
        #     markerfacecolor=tuple(col),
        #     markeredgecolor="k",
        #     markersize=14,
        # )

        ax.plot(
            x_pos[class_member_mask & core_samples_mask],
            y_pos[class_member_mask & core_samples_mask],
            z_pos[class_member_mask & core_samples_mask],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"{i} Estimated number of clusters: {n_clusters_}")
    plt.show()

# # KMEANS
# states = samples['avg_state']
# db = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(states)
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pos = pca.fit_transform(states)
# x_pos, y_pos, z_pos = pos[:, 0], pos[:, 1], pos[:, -1]
# # plt.scatter(x_pos, y_pos)
# # plt.show()
# #
# # ppl = {'x': x_pos, 'y': y_pos, 'label': labels}
# # pickle.dump(ppl, open(f'./data/echr_bert_pca.pkl', 'wb'))
#
# unique_labels = set(labels)
# fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot()
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = labels == k
#
#     ax.plot(
#         x_pos[class_member_mask],
#         y_pos[class_member_mask],
#         # z_pos[class_member_mask],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=6,
#     )
#
# # plt.title(f"Estimated number of clusters: {n_clusters_}")
# plt.show()
