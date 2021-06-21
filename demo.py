'''
Main file for creating simulated data or loading real data
and running RegGNN and sample selection methods.

Usage:
    For data processing:
        python demo.py --mode data --data-source SOURCE --measure SELECTION_METHOD

    For inferences:
        python demo.py --mode infer --model MODEL_NAME

    For more information:
        python demo.py -h
'''

import argparse
import pickle

import torch
import numpy as np

import proposed_method.data_utils as data_utils
import evaluators
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['data', 'infer'],
                    help="Creates data and topological features OR make inferences on data")

parser.add_argument('--model', default='RegGNN', choices=['CPM', 'PNA', 'RegGNN'],
                    help="Chooses the inference model that will be used")

parser.add_argument('--data-source', default='simulated', choices=['simulated', 'saved'],
                    help="Simulates random data or loads from path in config")

parser.add_argument('--measure', default='eigen',
                    choices=['abs', 'geo', 'tan', 'node', 'eigen', 'close', 'concat_orig', 'concat_scale'],
                    help="Chooses the topological measure to be used")

opts = parser.parse_args()

if opts.mode == 'data':
    '''
    Data is created or loaded according to the given params
    and topological features will be extracted from the data.

    Data and features are saved to the folder specified in config.py.
    '''

    print(f"'{opts.data_source}' data will be used with '{opts.measure}' measure.")
    if opts.data_source == 'simulated':
        conn, score = data_utils.simulate_dataset(conn_mean=Config.CONNECTOME_MEAN, conn_std=Config.CONNECTOME_STD,
                                                  score_mean=Config.SCORE_MEAN, score_std=Config.SCORE_STD,
                                                  n_subjects=Config.N_SUBJECTS, seed=Config.DATA_SEED)
        torch.save(conn, f"{Config.DATA_FOLDER}connectome.ts")
        torch.save(score, f"{Config.DATA_FOLDER}score.ts")
    elif opts.data_source == 'saved':
        conn = torch.load(f"{Config.DATA_FOLDER}connectome.ts")
        score = torch.load(f"{Config.DATA_FOLDER}score.ts")
    else:
        raise Exception("Unknown argument.")

    data_dict, score_dict = data_utils.create_features(conn.numpy(), score.numpy(), opts.measure)

    with open(f"{Config.DATA_FOLDER}data_dict.pkl", 'wb') as dd:
        pickle.dump(data_dict, dd)
    with open(f"{Config.DATA_FOLDER}score_dict.pkl", 'wb') as sd:
        pickle.dump(score_dict, sd)

    print(f"Data and topological features are created and saved at {Config.DATA_FOLDER} successfully.")

elif opts.mode == 'infer':
    '''
    Cross validation will be used to train and generate inferences
    on the data saved in the folder specified in config.py.

    Overall MAE and RMSE will be printed and predictions will be saved
    in same data folder.
    '''
    print(f"{opts.model} will be run on the data.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae_evaluator = lambda p, s: np.mean(np.abs(p - s))
    rmse_evaluator = lambda p, s: np.sqrt(np.mean((p - s) ** 2))

    if opts.model == 'CPM':
        preds, scores = evaluators.evaluate_CPM(seed=Config.MODEL_SEED,
                                                sample_selection=Config.SampleSelection.SAMPLE_SELECTION)
        if Config.SampleSelection.SAMPLE_SELECTION:
            mae_arr = [mae_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            rmse_arr = [rmse_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            print(f"For k in {Config.SampleSelection.K_LIST}:")
            print(f"Mean MAE +- std over k: {np.mean(mae_arr):.3f} +- {np.std(mae_arr):.3f}")
            print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")
            print(f"Mean RMSE +- std over k: {np.mean(rmse_arr):.3f} +- {np.std(rmse_arr):.3f}")
            print(f"Min, Max RMSE over k: {np.min(rmse_arr):.3f}, {np.max(rmse_arr):.3f}")
        else:
            print(f"MAE: {mae_evaluator(preds, scores):.3f}")
            print(f"RMSE: {rmse_evaluator(preds, scores):.3f}")
    elif opts.model == 'PNA':
        preds, scores, _ = evaluators.evaluate_PNA(shuffle=Config.SHUFFLE, random_state=Config.MODEL_SEED,
                                                   dropout=Config.PNA.DROPOUT, lr=Config.PNA.LR, wd=Config.PNA.WD,
                                                   num_epoch=Config.PNA.NUM_EPOCH, device=device,
                                                   sample_selection=Config.SampleSelection.SAMPLE_SELECTION,
                                                   k_list=Config.SampleSelection.K_LIST,
                                                   n_select_splits=Config.SampleSelection.N_SELECT_SPLITS)
        if Config.SampleSelection.SAMPLE_SELECTION:
            mae_arr = [mae_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            rmse_arr = [rmse_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            print(f"For k in {Config.SampleSelection.K_LIST}:")
            print(f"Mean MAE +- std over k: {np.mean(mae_arr):.3f} +- {np.std(mae_arr):.3f}")
            print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")
            print(f"Mean RMSE +- std over k: {np.mean(rmse_arr):.3f} +- {np.std(rmse_arr):.3f}")
            print(f"Min, Max RMSE over k: {np.min(rmse_arr):.3f}, {np.max(rmse_arr):.3f}")
        else:
            print(f"MAE: {mae_evaluator(preds, scores):.3f}")
            print(f"RMSE: {rmse_evaluator(preds, scores):.3f}")
    elif opts.model == 'RegGNN':
        preds, scores, _ = evaluators.evaluate_RegGNN(shuffle=Config.SHUFFLE, random_state=Config.MODEL_SEED,
                                                      dropout=Config.RegGNN.DROPOUT, k_list=Config.SampleSelection.K_LIST,
                                                      lr=Config.RegGNN.LR, wd=Config.RegGNN.WD, device=device,
                                                      sample_selection=Config.SampleSelection.SAMPLE_SELECTION,
                                                      num_epoch=Config.RegGNN.NUM_EPOCH,
                                                      n_select_splits=Config.SampleSelection.N_SELECT_SPLITS)
        if Config.SampleSelection.SAMPLE_SELECTION:
            mae_arr = [mae_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            rmse_arr = [rmse_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
            print(f"For k in {Config.SampleSelection.K_LIST}:")
            print(f"Mean MAE +- std over k: {np.mean(mae_arr):.3f} +- {np.std(mae_arr):.3f}")
            print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")
            print(f"Mean RMSE +- std over k: {np.mean(rmse_arr):.3f} +- {np.std(rmse_arr):.3f}")
            print(f"Min, Max RMSE over k: {np.min(rmse_arr):.3f}, {np.max(rmse_arr):.3f}")
        else:
            print(f"MAE: {mae_evaluator(preds, scores):.3f}")
            print(f"RMSE: {rmse_evaluator(preds, scores):.3f}")

    else:
        raise Exception("Unknown argument.")

    with open(f"{Config.RESULT_FOLDER}preds.pkl", 'wb') as f:
        pickle.dump(preds, f)
    with open(f"{Config.RESULT_FOLDER}scores.pkl", 'wb') as f:
        pickle.dump(scores, f)

    print(f"Predictions are successfully saved at {Config.RESULT_FOLDER}.")

else:
    raise Exception("Unknown argument.")
