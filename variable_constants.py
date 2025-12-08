## File to define the fixed constant(s) of the project

SHARON_IDX, SERGIO_IDX, MADDOX_IDX  = 0, 1, 2
OFFLINE_SESSION_IDX, ONLINE_SESSION_1_IDX, ONLINE_SESSION_2_IDX = 0, 1, 2

TRAIN_SUBJECT_IDX_TO_USE = SHARON_IDX
TRAIN_SESSION_IDX_TO_USE = OFFLINE_SESSION_IDX

TEST_SUBJECT_IDX_TO_USE = SHARON_IDX
TEST_SESSION_IDX_TO_USE = ONLINE_SESSION_2_IDX

DO_EXTRA_TRAIN_SESSION = True
if DO_EXTRA_TRAIN_SESSION:
    EXTRA_TRAIN_SUBJECT_IDX_TO_USE = SHARON_IDX
    EXTRA_TRAIN_SESSION_IDX_TO_USE = ONLINE_SESSION_1_IDX
else:
    EXTRA_TRAIN_SUBJECT_IDX_TO_USE = None
    EXTRA_TRAIN_SESSION_IDX_TO_USE = None

LIST_TYPE_SPATIAL_FILTER = ['none', 'car']
TYPE_SPATIAL_FILTER = LIST_TYPE_SPATIAL_FILTER[1]

POSSIBLE_CH_SELECTION_1 = ['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2']
POSSIBLE_CH_SELECTION_2 = ['C3', 'CP1', 'CP5', 'FC1']
ALL_POSSIBLE_CH_SELECTIONS = [POSSIBLE_CH_SELECTION_1, POSSIBLE_CH_SELECTION_2]
IDX_POSSIBLE_CH_SELECTION = 0
CH_SELECTION = ALL_POSSIBLE_CH_SELECTIONS[IDX_POSSIBLE_CH_SELECTION]

RANDOM_SEED = 42

# TODO implement leave one run out
LEAVE_ONE_RUN_INSTEAD_OF_EPOCH_OUT_CV = False

# Make sure feature and model is same group, model won't check this
DO_VALIDATION = True
NO_VALIDATION_FEATURE = 'group_a_welch_2_freq_bands_pca_8'
NO_VALIDATION_MODEL = 'group_a_lda'

LIST_FEATURE_SELECTION_OPTIONS = [
    'group_a_welch_2_freq_bands_no_pca', 'group_a_welch_3_freq_bands_no_pca', 'group_a_welch_small_freq_bands_no_pca',
    'group_a_welch_2_freq_bands_pca_4', 'group_a_welch_3_freq_bands_pca_4', 'group_a_welch_small_freq_bands_pca_4',
    'group_a_welch_2_freq_bands_pca_8', 'group_a_welch_3_freq_bands_pca_8', 'group_a_welch_small_freq_bands_pca_8',
    'group_a_csp_4_no_log', 'group_a_csp_8_no_log',
    'group_a_csp_4_log', 'group_a_csp_8_log',
    'group_b_cov_oas'
]
LIST_FEATURE_SELECTION_OPTIONS = [
    'group_b_cov_oas'
]

LIST_MODEL_OPTIONS = [
    'group_a_svm_linear', 'group_a_svm_rbf',
    'group_a_lda',
    'group_b_mdm_riem', 'group_b_mdm_euc'
]
LIST_MODEL_OPTIONS = [
    'group_b_mdm_riem', 'group_b_mdm_euc'
]

# -------

SAVE_PLOTS = True

DEBUG, INFO, WARNING, ERROR, CRITICAL = 10, 20, 30, 40, 50
LOGGER_LEVEL_OPTIONS = [DEBUG, INFO, WARNING, ERROR, CRITICAL]
LOGGER_LEVEL_CONSOLE = LOGGER_LEVEL_OPTIONS[1]
LOGGER_LEVEL_FILE = LOGGER_LEVEL_OPTIONS[1]

LIBRARIES_TO_IGNORE_LOGS_FROM = ['matplotlib', 'mne', 'numpy', 'pyxdf', 'sklearn']

RESAMPLE_SR = 256 # IN Hz

LOW_FREQ_BANDPASS = 8 # In Hz
HIGH_FREQ_BANDPASS = 30 # In Hz

LIST_CHANNELS_TO_REMOVE_BEFOREHAND = ['M1', 'M2', 'T7', 'T8']
REMOVE_SOME_CHANNELS_BEFOREHAND = True

LIST_EPOCH_REJECTION_CRITERIA = ['none']
EPOCH_REJECTION_CRITERIA = LIST_EPOCH_REJECTION_CRITERIA[0]

LIST_WINDOW_REJECTION_CRITERIA = ['none', 'ptp_150_microV']
WINDOW_REJECTION_CRITERIA = LIST_WINDOW_REJECTION_CRITERIA[1]

DURATION_WINDOW = 1 # In s
DT_WINDOW = 0.125 # In s

N_VALIDATION_FOLDS = 10

# This has to depend on the unique set of variables defining this run
# TODO check this has all info
LOG_FILE_NAME = f'train_{TRAIN_SUBJECT_IDX_TO_USE}_{TRAIN_SESSION_IDX_TO_USE}_test_{TEST_SUBJECT_IDX_TO_USE}_{TEST_SESSION_IDX_TO_USE}_extra_train_{EXTRA_TRAIN_SUBJECT_IDX_TO_USE}_{EXTRA_TRAIN_SESSION_IDX_TO_USE}_ch_list_{IDX_POSSIBLE_CH_SELECTION}_spatial_{TYPE_SPATIAL_FILTER}_val_{DO_VALIDATION}_leave_run_out_intead_of_epoch_{LEAVE_ONE_RUN_INSTEAD_OF_EPOCH_OUT_CV}_seed_{RANDOM_SEED}'

