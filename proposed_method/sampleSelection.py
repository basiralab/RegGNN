'''
Sample selection function.
'''

import pickle
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import networkx as nx

import SPD
import proposed_method.data_utils as data_utils

def select_samples(train_idx, n_splits, k_list,
                   data_dict, score_dict, shuffle, rs):
    '''Using the provided data and score dictionaries,
    selects the most important samples
    '''
    freq_dict = {k: defaultdict(int) for k in k_list}
    for fold, (train_ids, holdout_ids) in enumerate(KFold(n_splits=n_splits,
                                                          shuffle=shuffle,
                                                          random_state=rs).split(train_idx)):

        residuals = []
        fiq_errors = []

        for i in range(len(train_ids)):
            for j in range(i + 1, len(train_ids)):
                distance = data_dict[train_idx[train_ids[i]], train_idx[train_ids[j]]]
                fiq_error = score_dict[train_idx[train_ids[i]], train_idx[train_ids[j]]]
                residuals.append(distance)
                fiq_errors.append(fiq_error)

        if np.array(residuals[0]).shape == ():
            residuals = np.array(residuals).reshape(-1, 1)

        selectionNet = LinearRegression()
        selectionNet.fit(residuals, fiq_errors)

        for i in range(holdout_ids.shape[0]):
            R_tst = []
            for j in range(train_ids.shape[0]):
                distance = data_dict[train_idx[holdout_ids[i]], train_idx[train_ids[j]]]
                R_tst.append(distance)
            R_tst = np.stack(R_tst)
            if np.array(R_tst[0]).shape == ():
                R_tst = np.array(R_tst).reshape(-1, 1)
            error_pred = selectionNet.predict(R_tst).ravel()
            for k in k_list:
                for id in train_idx[train_ids[np.argsort(error_pred)[:k]]]:
                    freq_dict[k][id] += 1
    important_samples = {k: [] for k in k_list}
    for k, fd in freq_dict.items():
        ids, freqs = np.array(list(fd.keys())), np.array(list(fd.values()))
        important_samples[k] = ids[np.argsort(freqs)[::-1][:k]]
    return important_samples
