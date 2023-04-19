'''
Utility functions for data processing.
'''

import copy
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import torch
import torch_geometric
import SPD
import networkx as nx
from tqdm import tqdm

from config import Config


def simulate_dataset(conn_mean=0.0, conn_std=1.0,
                     score_mean=90.0, score_std=10.0,
                     n_subjects=30, seed=None):
    rng = np.random.default_rng(seed)
    conn = rng.normal(loc=conn_mean, scale=conn_std, size=(Config.ROI, Config.ROI, n_subjects))
    if Config.SPD:
        for i in range(n_subjects):
            conn[:, :, i] = conn[:, :, i] @ conn[:, :, i].T
    score = rng.normal(loc=score_mean, scale=score_std, size=(n_subjects,))
    return torch.tensor(conn), torch.tensor(score)


def create_features(data, score, method='eigen'):
    '''Given data matrix and score vector, creates and saves
    the dictionaries for pairwise similarity features.
    Possible values for method:
        'abs': absolute differences
        'geo': geometric distance
        'tan': vectorized tangent matrix
        'node': node degree centrality
        'eigen': eigenvector centrality
        'close': closeness centrality
    '''

    data_dict = {}
    score_dict = {}
    spd = SPD.SPD(d=Config.ROI)
    print("Starting topological feature extraction...")
    for i in tqdm(range(data.shape[2])):
        for j in range(i + 1, data.shape[2]):
            if method == 'abs':
                dist = np.abs(data[:, :, i] - data[:, :, j])
                dist = dist[np.triu_indices_from(dist, k=1)]
            if method == 'geo':
                dist = spd.dist(data[:, :, i], data[:, :, j])
            if method in ('tan', 'node', 'eigen', 'close', 'concat_orig', 'concat_scale'):
                dist = spd.transp(data[:, :, 0] + np.eye(Config.ROI) * 1e-10, data[:, :, i] + np.eye(Config.ROI) * 1e-10,
                                  spd.log(data[:, :, i] + np.eye(Config.ROI) * 1e-10, data[:, :, j] + np.eye(Config.ROI) * 1e-10))
                if method == 'tan':
                    dist = dist[np.triu_indices_from(dist)]
                if method == 'node':
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(list(dict(dist.degree(weight="weight")).values()))
                if method == 'eigen':
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(list(dict(nx.eigenvector_centrality(dist, weight="weight")).values()))
                if method == 'close':
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(list(dict(nx.closeness_centrality(dist, distance="weight")).values()))
                if method in ('concat_orig', 'concat_scale'):
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist_a = np.array(list(dict(dist.degree(weight="weight")).values()))
                    dist_b = np.array(list(dict(nx.eigenvector_centrality(dist, weight="weight")).values()))
                    dist_c = np.array(list(dict(nx.closeness_centrality(dist, distance="weight")).values()))
                    dist = (dist_a, dist_b, dist_c)
            data_dict[(i, j)] = data_dict[(j, i)] = dist
            score_dict[(i, j)] = score_dict[(j, i)] = np.abs(score[i] - score[j])
    if method == 'concat_orig':
        dicts = data_dict
        data_dict = {}
        for key in dicts.keys():
            data_dict[key] = np.concatenate(dicts[key])
    if method == 'concat_scale':
        dicts = data_dict
        lists = np.array([[x[0], x[1], x[2]] for x in dicts.values()])
        list_a = lists[:, 0]
        list_b = lists[:, 1]
        list_c = lists[:, 2]
        max_a, min_a = np.max(list_a, axis=0), np.min(list_a, axis=0)
        max_b, min_b = np.max(list_b, axis=0), np.min(list_b, axis=0)
        max_c, min_c = np.max(list_c, axis=0), np.min(list_c, axis=0)
        diff_a = max_a - min_a
        diff_b = max_b - min_b
        diff_c = max_c - min_c
        data_dict = {}
        for key in dicts.keys():
            a, b, c = dicts[key]
            data_dict[key] = np.concatenate(((a - min_a) / diff_a, (b - min_b) / diff_b, (c - min_c) / diff_c))
        
    return data_dict, score_dict


def load_dataset_pytorch():
    '''Loads the data for the given population into a list of Pytorch Geometric
    Data objects, which then can be used to create DataLoaders.
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectome.ts")
    scores = torch.load(f"{Config.DATA_FOLDER}score.ts")

    connectomes[connectomes < 0] = 0

    pyg_data = []
    for subject in range(scores.shape[0]):
        sparse_mat = to_sparse(connectomes[:, :, subject])
        pyg_data.append(torch_geometric.data.Data(x=torch.eye(Config.ROI, dtype=torch.float),
                                                  y=scores[subject].float(), edge_index=sparse_mat._indices(),
                                                  edge_attr=sparse_mat._values().float()))

    return pyg_data


def to_sparse(mat):
    '''Transforms a square matrix to torch.sparse tensor

    Methods ._indices() and ._values() can be used to access to
    edge_index and edge_attr while generating Data objects
    '''
    coo = coo_matrix(mat, dtype='float64')
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    coo_index = torch.stack([row, col], dim=0)
    coo_values = torch.from_numpy(coo.data.astype(np.float64).reshape(-1, 1)).reshape(-1)
    sparse_mat = torch.sparse.LongTensor(coo_index, coo_values)
    return sparse_mat


def load_dataset_cpm():
    '''Loads the data for given population in the upper triangular matrix form
    as required by CPM functions.
    '''
    connectomes = np.array(torch.load(f"{Config.DATA_FOLDER}connectome.ts"))
    scores = np.array(torch.load(f"{Config.DATA_FOLDER}score.ts"))

    fc_data = {}
    behav_data = {}
    for subject in range(scores.shape[0]):  # take upper triangular part of each matrix
        fc_data[subject] = connectomes[:, :, subject][np.triu_indices_from(connectomes[:, :, subject], k=1)]
        behav_data[subject] = {'score': scores[subject].item()}
    return pd.DataFrame.from_dict(fc_data, orient='index'), pd.DataFrame.from_dict(behav_data, orient='index')


def get_loaders(train, test, batch_size=1):
    '''Returns data loaders for given data lists
    '''
    train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def load_dataset_tensor(pop="NT"):
    '''Loads dataset as tuple of (tensor of connectomes,
       tensor of fiq scores, tensor of viq scores)
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectome.ts")
    scores = torch.load(f"{Config.DATA_FOLDER}score.ts")
    return connectomes, scores


def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data
