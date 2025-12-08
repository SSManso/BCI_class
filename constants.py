## File to define the constant(s) subject to change of the project

PATH_LOGS_FOLDER = './logs/'
LOG_END_FILE = '.log'

PATH_PLOTS_FOLDER = './plots/'
PLOT_END_FILE = '.pdf'

PATH_DATA_FOLDER = './data/'
DATA_END_FILE = '.xdf'

SHARON = [
    'sub-F25CLASS_SUBJ_007_ses-S001OFFLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_007_ses-S001ONLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_007_ses-S002ONLINE_task-Default_run-001_eeg',
]
SERGIO = [
    'sub-F25CLASS_SUBJ_008_ses-S001OFFLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_008_ses-S002ONLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_008_ses-S003ONLINE_task-Default_run-001_eeg'
]
MADDOX = [
    'sub-LAB_SUBJ_001_ses-S007OFFLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_009_ses-S001ONLINE_task-Default_run-001_eeg',
    'sub-F25CLASS_SUBJ_009_ses-S002ONLINE_task-Default_run-001_eeg'
]
SUBJECTS = [SHARON, SERGIO, MADDOX]
SUBJECTS_NAME = ['Sharon', 'Sergio', 'Maddox']
SESSIONS_NAME = ['Offline', 'Online 1', 'Online 2s']

XDF_DEJITTER = 'True'
XDF_SYNC = 'True'

STREAM_INFO_NAME = 'info'
STREAM_TYPE_NAME = 'type'
STREAM_MARKER_NAME = 'Markers'
STREAM_EEG_NAME = 'EEG'

INITIAL_LIST_CH_TO_REMOVE = ['AUX1', 'AUX2', 'AUX3', 'AUX7', 'AUX8', 'AUX9', 'TRIGGER']

STREAM_TIME_SERIES_NAME = 'time_series'
STREAM_TIME_STAMPS_NAME = 'time_stamps'
STREAM_DESC_NAME = 'desc'
STREAM_CH_NAME_1 = 'channels'
STREAM_CH_NAME_2 = 'channel'
STREAM_CH_LABEL_NAME = 'label'
STREAM_NOMINAL_SR = 'nominal_srate'
STREAM_EFFECTIVE_SR = 'effective_srate'

STREAM_MARKER_VALUE_IDX = 0
STREAM_MARKER_TIME_STAMP_IDX = 1

TRIGGERS_KEYS = {
    100: 'REST_BEGIN',
    120: 'REST_END',
    200: 'MI_BEGIN',
    220: 'MI_END'
}
MI_NAME, REST_NAME, BEGIN_NAME, END_NAME = 'MI', 'REST', 'BEGIN', 'END'
MI_LABEL, REST_LABEL = 1, 0

MNE_STANDARD_MONTAGE = 'standard_1020'
MNE_SR = 'sfreq'

MICROVOLTS_TO_VOLTS = 1e-6
SEC_TO_MIN = 1 / 60

FILTER_FREQ_BANDPASS_ORDER = 4
FILTER_FREQ_NOTCH_ORDER = 4
FILTER_FREQ_NOTCH_CENTER = 60 # In Hz
FILTER_FREQ_NOTCH_HALF_WINDOW = 2 # In Hz

SPATIAL_FILTER_NONE = 'none'
SPATIAL_FILTER_CAR = 'car'

ANNOTATIONS_ONSET_NAME = 'onset'
ANNOTATIONS_TYPE_NAME = 'description'

DO_EPOCH_BASELINE_CORRECTION = True
EPOCH_BASELINE_CORRECTION_DURATION = 0.2 # In s

WINDOW_REJECTION_NONE = 'none'
WINDOW_REJECTION_PTP_150_MICROV = 'ptp_150_microV'

WINDOW_REJECTION_PTP_150_MICROV_THRESHOLD = 150e-6 # In V

FEATURE_SELECTION_WELCH_GENERAL = 'welch'
FEATURE_SELECTION_WELCH_2_NO_PCA = 'group_a_welch_2_freq_bands_no_pca'
FEATURE_SELECTION_WELCH_3_NO_PCA = 'group_a_welch_3_freq_bands_no_pca'
FEATURE_SELECTION_WELCH_SMALL_NO_PCA = 'group_a_welch_small_freq_bands_no_pca'
FEATURE_SELECTION_WELCH_2_PCA_4 = 'group_a_welch_2_freq_bands_pca_4'
FEATURE_SELECTION_WELCH_3_PCA_4 = 'group_a_welch_3_freq_bands_pca_4'
FEATURE_SELECTION_WELCH_SMALL_PCA_4 = 'group_a_welch_small_freq_bands_pca_4'
FEATURE_SELECTION_WELCH_2_PCA_8 = 'group_a_welch_2_freq_bands_pca_8'
FEATURE_SELECTION_WELCH_3_PCA_8 = 'group_a_welch_3_freq_bands_pca_8'
FEATURE_SELECTION_WELCH_SMALL_PCA_8 = 'group_a_welch_small_freq_bands_pca_8'

FEATURE_SELECTION_CSP_GENERAL = 'csp'
FEATURE_SELECTION_CSP_4_NO_LOG = 'group_a_csp_4_no_log'
FEATURE_SELECTION_CSP_8_NO_LOG = 'group_a_csp_8_no_log'
FEATURE_SELECTION_CSP_4_LOG = 'group_a_csp_4_log'
FEATURE_SELECTION_CSP_8_LOG = 'group_a_csp_8_log'

FEATURE_SELECTION_COV_GENERAL = 'cov'
FEATURE_SELECTION_COV_OAS = 'group_b_cov_oas'

# In Hz
WELCH_FREQ_BANDS_2 = [(8, 13), (13, 30)]
WELCH_FREQ_BANDS_3 = [(8, 13), (13, 20), (20, 30)]
WELCH_FREQ_BANDS_SMALL = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30)]

MODEL_SVM_LINEAR = 'group_a_svm_linear'
MODEL_SVM_RBF = 'group_a_svm_rbf'

MODEL_LDA = 'group_a_lda'

MODEL_MDM_RIEM = 'group_b_mdm_riem'
MODEL_MDM_EUC = 'group_b_mdm_euc'

METRIC_ACC = "acc"
METRIC_PREC = "precision"
METRIC_REC = "recall"
METRIC_F1 = "f1"
METRIC_BAL_ACC = "balanced_acc"
METRIC_CONFMAT = "confusion_matrix"
METRIC_TPR = "tpr"       # sensitivity / recall
METRIC_TNR = "tnr"       # specificity
METRIC_FPR = "fpr"
METRIC_FNR = "fnr"
METRIC_TP  = "tp"
METRIC_TN  = "tn"
METRIC_FP  = "fp"
METRIC_FN  = "fn"

N_DECIMALS = 4

N_ITERATIONS_RANDOM_CHANCE = 1000