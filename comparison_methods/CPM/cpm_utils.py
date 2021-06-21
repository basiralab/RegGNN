'''Functions for performing CPM on connectome data.

Originally created by github:esfinn at https://github.com/esfinn/cpm_tutorial/
'''


from random import sample
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import KFold

def mk_kfold_indices(subj_list, k=10):
    """
    Splits list of subjects into k folds for cross-validation.
    """

    n_subs = len(subj_list)
    n_subs_per_fold = n_subs // k  # floor integer for n_subs_per_fold

    indices = [[fold_no] * n_subs_per_fold for fold_no in range(k)]  # generate repmat list of indices
    remainder = n_subs % k  # figure out how many subs are left over
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]
    [indices.append(ind) for ind in remainder_inds]  # add indices for remainder subs

    assert len(indices) == n_subs, "Length of indices list does not equal number of subjects, something went wrong"

    indices = np.array(indices)
    i = 0
    for train_inds, test_inds in KFold(k, shuffle=True, random_state=1).split(subj_list):
      indices[test_inds] = i
      i += 1
    return np.array(indices)


def split_train_test(subj_list, indices, test_fold):
    """
    For a subj list, k-fold indices, and given fold, returns lists of train_subs and test_subs
    """

    train_inds = np.where(indices != test_fold)
    test_inds = np.where(indices == test_fold)

    train_subs = []
    for sub in subj_list[train_inds]:
        train_subs.append(sub)

    test_subs = []
    for sub in subj_list[test_inds]:
        test_subs.append(sub)

    return (train_subs, test_subs)


def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):
    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]

    return (train_vcts, train_behav, test_vcts)


def select_features(train_vcts, train_behav, r_thresh=0.2, corr_type='pearson', verbose=False):
    """
    Runs the CPM feature selection step:
    - correlates each edge with behavior, and returns a mask of edges that are correlated above some threshold, one for each tail (positive and negative)
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type == 'pearson':
        cov = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0] - 1)
        corr = cov / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
    elif corr_type == 'spearman':
        corr = []
        for edge in train_vcts.columns:
            r_val = sp.stats.spearmanr(train_vcts.loc[:, edge], train_behav)[0]
            corr.append(r_val)

    # Define positive and negative masks
    mask_dict = {}
    mask_dict["pos"] = corr > r_thresh
    mask_dict["neg"] = corr < -r_thresh

    if verbose:
        print("Found ({}/{}) edges positively/negatively correlated with behavior in the training set".format(mask_dict["pos"].sum(), mask_dict["neg"].sum()))  # for debugging

    return mask_dict


def build_model(train_vcts, mask_dict, train_behav):
    """
    Builds a CPM model:
    - takes a feature mask, sums all edges in the mask for each subject, and uses simple linear regression to relate summed network strength to behavior
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    model_dict = {}

    # Loop through pos and neg tails
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.items())))

    t = 0
    for tail, mask in mask_dict.items():
        X = train_vcts.values[:, mask].sum(axis=1)
        X_glm[:, t] = X
        y = train_behav
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t += 1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict


def apply_model(test_vcts, mask_dict, model_dict):
    """
    Applies a previously trained linear regression model to a test set to generate predictions of behavior.
    """

    behav_pred = {}

    X_glm = np.zeros((test_vcts.shape[0], len(mask_dict.items())))

    # Loop through pos and neg tails
    t = 0
    for tail, mask in mask_dict.items():
        X = test_vcts.loc[:, mask].sum(axis=1)
        X_glm[:, t] = X

        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope * X + intercept
        t += 1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    behav_pred["glm"] = np.dot(X_glm, model_dict["glm"])

    return behav_pred


def cpm_wrapper(all_fc_data, all_behav_data, behav, k=10,
                sample_selection=False, sample_atlas=None, k_list=[0], **cpm_kwargs):

    assert all_fc_data.index.equals(all_behav_data.index), "Row (subject) indices of FC vcts and behavior don't match!"

    results = {}
    scores = {}
    for k_ in k_list:

        subj_list = all_fc_data.index  # get subj_list from df index

        indices = mk_kfold_indices(subj_list, k=k)

        # Initialize df for storing observed and predicted behavior
        col_list = []
        for tail in ["pos", "neg", "glm"]:
            col_list.append(behav + " predicted (" + tail + ")")
        col_list.append(behav + " observed")
        behav_obs_pred = pd.DataFrame(index=subj_list, columns=col_list)

        # Initialize array for storing feature masks
        n_edges = all_fc_data.shape[1]
        all_masks = {}
        all_masks["pos"] = np.zeros((k, n_edges))
        all_masks["neg"] = np.zeros((k, n_edges))

        for fold in range(k):
            print("doing fold {}".format(fold))
            train_subs, test_subs = split_train_test(subj_list, indices, test_fold=fold)
            if sample_selection:
                train_subs = sample_atlas[fold][k_]
            train_vcts, train_behav, test_vcts = get_train_test_data(all_fc_data, train_subs, test_subs, all_behav_data, behav=behav)
            mask_dict = select_features(train_vcts, train_behav, **cpm_kwargs)
            all_masks["pos"][fold, :] = mask_dict["pos"]
            all_masks["neg"][fold, :] = mask_dict["neg"]
            model_dict = build_model(train_vcts, mask_dict, train_behav)
            behav_pred = apply_model(test_vcts, mask_dict, model_dict)
            for tail, predictions in behav_pred.items():
                behav_obs_pred.loc[test_subs, behav + " predicted (" + tail + ")"] = predictions

        behav_obs_pred.loc[subj_list, behav + " observed"] = all_behav_data[behav]
        cpm_preds, cpm_scores = behav_obs_pred[f"{behav} predicted (glm)"], behav_obs_pred[f"{behav} observed"]
        cpm_preds = cpm_preds.to_numpy(dtype=np.float64).ravel()
        cpm_scores = cpm_scores.to_numpy(dtype=np.float64).ravel()

        results[k_] = cpm_preds
        scores[k_] = cpm_scores

    if k_list == [0]:
        results = results[0]
        scores = scores[0]

    return results, scores
