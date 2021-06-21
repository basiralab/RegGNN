'''
Configuration options for RegGNN pipeline.
'''


class Config:
    # SYSTEM OPTIONS
    DATA_FOLDER = './simulated_data/'  # path to the folder data will be written to and read from
    RESULT_FOLDER = './'  # path to the folder data will be written to and read from

    # SIMULATED DATA OPTIONS
    CONNECTOME_MEAN = 0.0  # mean of the distribution from which connectomes will be sampled
    CONNECTOME_STD = 1.0  # std of the distribution from which connectomes will be sampled
    SCORE_MEAN = 90.0  # mean of the distribution from which scores will be sampled
    SCORE_STD = 10.0  # std of the distribution from which scores will be sampled
    N_SUBJECTS = 30  # number of subjects in the simulated data
    ROI = 116  # number of regions of interest in brain graph
    SPD = True  # whether or not to make generated matrices symmetric positive definite

    # EVALUATION OPTIONS
    K_FOLDS = 5  # number of cross validation folds

    # REGGNN OPTIONS
    class RegGNN:
        NUM_EPOCH = 100  # number of epochs the process will be run for
        LR = 1e-3  # learning rate
        WD = 5e-4  # weight decay
        DROPOUT = 0.1  # dropout rate

    # PNA OPTIONS
    class PNA:
        NUM_EPOCH = 100  # number of epochs the process will be run for
        LR = 1e-4  # learning rate
        WD = 5e-4  # weight decay
        DROPOUT = 0.1  # dropout rate
        SCALERS = ["identity", "amplification", "attenuation"]  # scalers used by PNA
        AGGRS = ['sum', 'mean', 'var', 'max']  # aggregators used by PNA

    class SampleSelection:
        SAMPLE_SELECTION = True  # whether or not to apply sample selection
        K_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # list of k values for sample selection
        N_SELECT_SPLITS = 10  # number of folds for the nested sample selection cross validation

    # RANDOMIZATION OPTIONS
    DATA_SEED = 1  # random seed for data creation
    MODEL_SEED = 1  # random seed for models
    SHUFFLE = True  # whether to shuffle or not
