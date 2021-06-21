'''
Functions for k-fold evaluation of models.
'''

import random
import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch

import proposed_method.data_utils as data_utils
import comparison_methods.CPM.cpm_utils as cpm_utils
import proposed_method.sampleSelection as sampsel
from proposed_method.RegGNN import RegGNN
from comparison_methods.PNA.model import PNANet, compute_deg

from config import Config


def evaluate_CPM(seed, sample_selection):
    '''Evaluates CPM on population pop in ("NT" or "ASD")
    and for score in ('fiq', 'viq'), uses seed for randomization
    '''
    connectomes, scores = data_utils.load_dataset_cpm()
    random.seed(seed)
    np.random.seed(seed)
    if sample_selection is True:
        with open(f"{Config.DATA_FOLDER}data_dict.pkl", 'rb') as dd:
            data_dict = pickle.load(dd)
        with open(f"{Config.DATA_FOLDER}score_dict.pkl", 'rb') as sd:
            score_dict = pickle.load(sd)
        data = data_utils.load_dataset_pytorch()
        sample_atlas = {}
        for i, (train_idx, test_idx) in enumerate(KFold(Config.K_FOLDS, shuffle=Config.SHUFFLE,
                                                        random_state=Config.MODEL_SEED).split(data)):
            sample_atlas[i] = sampsel.select_samples(train_idx, Config.SampleSelection.N_SELECT_SPLITS, 
                                                     Config.SampleSelection.K_LIST,
                                                     data_dict, score_dict, Config.SHUFFLE,
                                                     Config.MODEL_SEED)

        cpm_preds, cpm_scores = cpm_utils.cpm_wrapper(connectomes, scores, 'score', k=Config.K_FOLDS,
                                                      sample_atlas=sample_atlas, sample_selection=True,
                                                      k_list=Config.SampleSelection.K_LIST)
    else:
        cpm_preds, cpm_scores = cpm_utils.cpm_wrapper(connectomes, scores, 'score', k=Config.K_FOLDS)
    return cpm_preds, cpm_scores


def evaluate_RegGNN(sample_selection=False, shuffle=False, random_state=None,
                    dropout=0.1, k_list=list(range(2, 16)), lr=1e-3, wd=5e-4,
                    device=torch.device('cpu'), num_epoch=100, n_select_splits=10):

    if sample_selection is False:
        k_list = [0]

    if sample_selection:
        with open(f"{Config.DATA_FOLDER}data_dict.pkl", 'rb') as dd:
            data_dict = pickle.load(dd)
        with open(f"{Config.DATA_FOLDER}score_dict.pkl", 'rb') as sd:
            score_dict = pickle.load(sd)

    overall_preds = {k: [] for k in k_list}
    overall_scores = {k: [] for k in k_list}
    train_mae = {k: [] for k in k_list}

    data = data_utils.load_dataset_pytorch()
    fold = -1
    for train_idx, test_idx in KFold(Config.K_FOLDS, shuffle=shuffle,
                                     random_state=random_state).split(data):
        fold += 1
        print(f"Cross Validation Fold {fold+1}/{Config.K_FOLDS}")
        if sample_selection:
            sample_atlas = sampsel.select_samples(train_idx, n_select_splits, k_list,
                                                  data_dict, score_dict, shuffle,
                                                  random_state)

        for k in k_list:
            if sample_selection:
                selected_train_data = [data[subject] for subject in sample_atlas[k]]
            else:
                selected_train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            candidate_model = RegGNN(116, 64, 1, dropout).float().to(device)
            optimizer = torch.optim.Adam(candidate_model.parameters(), lr=lr, weight_decay=wd)
            train_loader, test_loader = data_utils.get_loaders(selected_train_data, test_data)
            candidate_model.train()
            for epoch in range(num_epoch):
                preds = []
                scores = []
                for batch in train_loader:
                    out = candidate_model(batch.x.to(device), data_utils.to_dense(batch).adj.to(device))
                    loss = candidate_model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))

                    candidate_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.long().numpy())

                preds = np.hstack(preds)
                scores = np.hstack(scores)
                epoch_mae = np.mean(np.abs(preds.reshape(-1, 1) - scores.reshape(-1, 1)))
                train_mae[k].append(epoch_mae)

            candidate_model.eval()
            with torch.no_grad():
                preds = []
                scores = []
                for batch in test_loader:
                    out = candidate_model(batch.x.to(device), data_utils.to_dense(batch).adj.to(device))

                    loss = candidate_model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.cpu().long().numpy())

                preds = np.hstack(preds)
                scores = np.hstack(scores)

            overall_preds[k].extend(preds)
            overall_scores[k].extend(scores)

    for k in k_list:
        overall_preds[k] = np.vstack(overall_preds[k]).ravel()
        overall_scores[k] = np.vstack(overall_scores[k]).ravel()

    if sample_selection is False:
        overall_preds = overall_preds[k_list[0]]
        overall_scores = overall_scores[k_list[0]]

    return overall_preds, overall_scores, train_mae


def evaluate_PNA(sample_selection=False, shuffle=False, random_state=None,
                 dropout=0.1, k_list=list(range(2, 16)), lr=1e-4, wd=5e-4,
                 device=torch.device('cpu'), num_epoch=100, n_select_splits=10):

    if sample_selection is False:
        k_list = [0]

    if sample_selection:
        with open(f"{Config.DATA_FOLDER}data_dict.pkl", 'rb') as dd:
            data_dict = pickle.load(dd)
        with open(f"{Config.DATA_FOLDER}score_dict.pkl", 'rb') as sd:
            score_dict = pickle.load(sd)

    overall_preds = {k: [] for k in k_list}
    overall_scores = {k: [] for k in k_list}
    train_mae = {k: [] for k in k_list}

    data = data_utils.load_dataset_pytorch()
    fold = -1
    for train_idx, test_idx in KFold(Config.K_FOLDS, shuffle=shuffle,
                                     random_state=random_state).split(data):
        fold += 1
        print(f"Cross Validation Fold {fold+1}/{Config.K_FOLDS}")

        if sample_selection:
            sample_atlas = sampsel.select_samples(train_idx, n_select_splits, k_list,
                                                  data_dict, score_dict, shuffle,
                                                  random_state)

        for k in k_list:
            if sample_selection:
                selected_train_data = [data[subject] for subject in sample_atlas[k]]
            else:
                selected_train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            train_loader, test_loader = data_utils.get_loaders(selected_train_data, test_data)
            candidate_model = PNANet(116, 64, 1, dropout, aggrs=Config.PNA.AGGRS, scalers=Config.PNA.SCALERS,
                                     deg=compute_deg(train_loader)).float().to(device)
            optimizer = torch.optim.Adam(candidate_model.parameters(), lr=lr, weight_decay=wd)
            candidate_model.train()
            for epoch in range(num_epoch):
                preds = []
                scores = []
                for batch in train_loader:
                    out = candidate_model(batch.x.to(device),
                                          batch.edge_index.to(device),
                                          batch.edge_attr.abs().to(device),
                                          batch=1)
                    loss = candidate_model.loss(out.view(-1, 1), batch.y.to(device).view(-1, 1))

                    candidate_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.long().numpy())
                preds = np.hstack(preds)
                scores = np.hstack(scores)
                epoch_mae = np.mean(np.abs(preds.reshape(-1, 1) - scores.reshape(-1, 1)))
                train_mae[k].append(epoch_mae)

            candidate_model.eval()
            with torch.no_grad():
                preds = []
                scores = []
                for batch in test_loader:
                    out = candidate_model(batch.x.to(device),
                                          batch.edge_index.to(device),
                                          batch.edge_attr.abs().to(device),
                                          batch=1)

                    loss = candidate_model.loss(out.view(-1, 1), batch.to(device).y.view(-1, 1))
                    preds.append(out.cpu().data.numpy())
                    scores.append(batch.y.cpu().long().numpy())

                preds = np.hstack(preds)
                scores = np.hstack(scores)
            overall_preds[k].extend(preds)
            overall_scores[k].extend(scores)

    for k in k_list:
        overall_preds[k] = np.hstack(overall_preds[k]).ravel()
        overall_scores[k] = np.hstack(overall_scores[k]).ravel()

    if sample_selection is False:
        overall_preds = overall_preds[k_list[0]]
        overall_scores = overall_scores[k_list[0]]

    return overall_preds, overall_scores, train_mae
