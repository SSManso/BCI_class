## File to define the main function of the project


## Imports

# From this project (constants)
from utils.variable_constants import LOGGER_LEVEL_CONSOLE, LOGGER_LEVEL_FILE, LOG_FILE_NAME, LIBRARIES_TO_IGNORE_LOGS_FROM
from utils.variable_constants import TRAIN_SUBJECT_IDX_TO_USE, TRAIN_SESSION_IDX_TO_USE, TEST_SUBJECT_IDX_TO_USE, TEST_SESSION_IDX_TO_USE
from utils.variable_constants import RESAMPLE_SR, LOW_FREQ_BANDPASS, HIGH_FREQ_BANDPASS, TYPE_SPATIAL_FILTER, REMOVE_SOME_CHANNELS_BEFOREHAND
from utils.variable_constants import EPOCH_REJECTION_CRITERIA
from utils.variable_constants import N_CHANNELS_TO_KEEP
from utils.variable_constants import DURATION_WINDOW, DT_WINDOW
from utils.variable_constants import N_VALIDATION_FOLDS
from utils.variable_constants import LIST_FEATURE_SELECTION_OPTIONS, LIST_MODEL_OPTIONS
# from utils.constants import CSP_N_COMPONENTS
from utils.constants import METRIC_ACC
from utils.constants import N_DECIMALS
from utils.constants import SUBJECTS_NAME, SESSIONS_NAME

# From this project (functions)
from utils.create_logger import create_logger
from utils.preprocessing import read_file_and_create_mne_with_annotations
from utils.preprocessing import resample_and_filter_mne_frequency_and_spatially
from utils.preprocessing import characterization_and_channel_selection
from utils.preprocessing import filter_channels_mne
from utils.preprocessing import epoch_mne
from utils.preprocessing import window_epochs_np_and_get_and_apply_rejection_criteria
from utils.preprocessing import window_epochs_np_and_apply_rejection_criteria
from utils.preprocessing import create_folds_np
from utils.preprocessing import feature_selection
from utils.preprocessing import shuffle_samples
from utils.decoder import create_model
from utils.decoder import train_model
from utils.decoder import test_model
from utils.decoder import average_list_metrics
from utils.decoder import max_list_metrics

# From sys library
from datetime import datetime
import logging
import time

# From external libraries
# - None


## Main pipeline (train on 1 session and test on 1 session)
# TODO organize for loops for hyperparameter searching aside of feature selection and models (e.g. window creation)
# TODO set seeds on everything so this is reproducible, this needs an external loop of several seeds to reduce bias from seed selection

def main():
    """
    Main function of the file

    Parameters:
        None

    Returns:
        None
    """

    # Set up logger
    create_logger(LOGGER_LEVEL_CONSOLE, LOGGER_LEVEL_FILE, LOG_FILE_NAME, LIBRARIES_TO_IGNORE_LOGS_FROM)
    logger = logging.getLogger(__name__)
    logger.info(f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")} - Starting run with unique ID [{LOG_FILE_NAME}]')

    # Read data
    logger.info('-- Train -- Read data')
    train_raw_mne = read_file_and_create_mne_with_annotations(TRAIN_SUBJECT_IDX_TO_USE, TRAIN_SESSION_IDX_TO_USE)
    logger.info('-- Test -- Read data')
    test_raw_mne = read_file_and_create_mne_with_annotations(TEST_SUBJECT_IDX_TO_USE, TEST_SESSION_IDX_TO_USE)

    # Filter data (freq (bp + notch) and spatially)
    logger.info('-- Train -- Filter data')
    train_filt_mne = resample_and_filter_mne_frequency_and_spatially(train_raw_mne, RESAMPLE_SR, LOW_FREQ_BANDPASS, HIGH_FREQ_BANDPASS, TYPE_SPATIAL_FILTER, REMOVE_SOME_CHANNELS_BEFOREHAND)
    logger.info('-- Test -- Filter data')
    test_filt_mne = resample_and_filter_mne_frequency_and_spatially(test_raw_mne, RESAMPLE_SR, LOW_FREQ_BANDPASS, HIGH_FREQ_BANDPASS, TYPE_SPATIAL_FILTER, REMOVE_SOME_CHANNELS_BEFOREHAND)

    # Characterization and channel selection based on train data
    logger.info('-- Train -- Channel selection')
    # list_channels_to_use = characterization_and_channel_selection(train_filt_mne, N_CHANNELS_TO_KEEP)
    logger.warning('See which list of electrodes to use')
    list_channels_to_use = ['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2']
    list_channels_to_use = ['C3', 'CP1', 'CP5', 'FC1']
    logger.info(f'Chosen channels: {list_channels_to_use}')
    train_clean_ch_filt_mne = filter_channels_mne(train_filt_mne, list_channels_to_use)
    test_clean_ch_filt_mne = filter_channels_mne(test_filt_mne, list_channels_to_use)

    # Epoch data (full trial)
    logger.info('-- Train -- Epoch data')
    train_epoch_np, train_epoch_labels = epoch_mne(train_clean_ch_filt_mne, EPOCH_REJECTION_CRITERIA)
    logger.info('-- Test -- Epoch data')
    test_epoch_np, test_epoch_labels = epoch_mne(test_clean_ch_filt_mne, EPOCH_REJECTION_CRITERIA)

    # Create samples for model (small windows)
    logger.info('-- Train -- Windowing epochs and get rejection criteria')
    train_windows_np, final_train_labels, final_train_epochs, window_rejection_criteria = window_epochs_np_and_get_and_apply_rejection_criteria(train_epoch_np, train_epoch_labels, RESAMPLE_SR, DURATION_WINDOW, DT_WINDOW)
    logger.info('-- Test -- Windowing epochs')
    test_windows_np, final_test_labels, final_test_epochs = window_epochs_np_and_apply_rejection_criteria(test_epoch_np, test_epoch_labels, RESAMPLE_SR, DURATION_WINDOW, DT_WINDOW, window_rejection_criteria)

    logger.warning('Try welch with more freq bands')
    logger.warning('Right now welch only does 1 window (256 points) and does 8-13 and 13-30 Hz')
    logger.warning('Right now CSP chooses 4 components and does log')
    logger.warning('Right now SVM uses a linear kernel')
    logger.warning('Right now metric is just per-window accuracy on all samples')
    logger.warning('Where should be the seed=Random seed, inside the random function or outside?')
    logger.warning('create cov + mdm')

    # Grid search + validation (k-fold) for best feature selection + model (ensure windows from the same trial fall in the same fold!)
    logger.info('-- Validation -- Create folds')
    list_folds_train_windows_np = create_folds_np(train_windows_np, final_train_labels, final_train_epochs, N_VALIDATION_FOLDS)
    all_combinations = [(each_feature_selection_option, each_model_option) for each_feature_selection_option in LIST_FEATURE_SELECTION_OPTIONS for each_model_option in LIST_MODEL_OPTIONS]
    
    logger.warning('Remove combinations that dont make sense')
    logger.info(f'All possible comnbinations: {all_combinations}')
    
    validation_metric = []
        # - Grid search loop
    for this_feature_selection_option, this_model_option in all_combinations:
        this_combination_metric = []

            # - k-fold loop
        logger.info(f'-- Validation -- k-fool loop for combination [feature - {this_feature_selection_option} || model - {this_model_option}]')
        t_0 = time.time()
        for train_fold_np, val_fold_np, train_labels, val_labels in list_folds_train_windows_np:

            # Feature selection
            train_input_model, val_input_model = feature_selection(train_fold_np, val_fold_np, this_feature_selection_option, RESAMPLE_SR, train_labels)

            # Shuffle samples
            train_input_model, train_labels = shuffle_samples(train_input_model, train_labels)
            val_input_model, val_input_model = shuffle_samples(val_input_model, val_input_model)

            # Train and validate model
            raw_model = create_model(this_model_option)
            trained_model, train_metric = train_model(raw_model, train_input_model, train_labels)
            val_metric = test_model(trained_model, val_input_model, val_labels)
            this_combination_metric.append(val_metric)
        
        # Save result
        validation_metric.append(average_list_metrics(this_combination_metric))
        logger.info(f'Val performance: {validation_metric[-1][METRIC_ACC]:.{N_DECIMALS}f}')

        t_f = time.time()
        logger.info(f'Completed in {int(round(t_f - t_0))} seconds')

    # Test final model
    logger.info('-- Test -- Evaluate final model')
    t_0 = time.time()
        # - Feature selection
    final_feature_selection_option, final_model_option = all_combinations[validation_metric.index(max_list_metrics(validation_metric))]
    train_input_model, test_input_model = feature_selection(train_windows_np, test_windows_np, final_feature_selection_option, RESAMPLE_SR, final_train_labels)
        # - Shuffle samples
    train_input_model, final_train_labels = shuffle_samples(train_input_model, final_train_labels)
    test_input_model, final_test_labels = shuffle_samples(test_input_model, final_test_labels)
        # - Train and test model
    raw_model = create_model(final_model_option)
    trained_model, train_metric = train_model(raw_model, train_input_model, final_train_labels)
    final_test_metric = test_model(trained_model, test_input_model, final_test_labels)

    t_f = time.time()
    logger.info(f'Completed in {int(round(t_f - t_0))} seconds')

    logger.info('-- Final results --')
    logger.info(f'Train session: {SUBJECTS_NAME[TRAIN_SUBJECT_IDX_TO_USE]} - {SESSIONS_NAME[TRAIN_SESSION_IDX_TO_USE]}')
    logger.info(f'Test session: {SUBJECTS_NAME[TEST_SUBJECT_IDX_TO_USE]} - {SESSIONS_NAME[TEST_SESSION_IDX_TO_USE]}')
    logger.info(f'Feature selection type: {final_feature_selection_option}')
    logger.info(f'Model type: {final_model_option}')
    logger.info(f'Final performance: {final_test_metric[METRIC_ACC]:.{N_DECIMALS}f}')

    logger.warning('feature importance heatmap, kinda like shapley?')
    logger.warning('report chance level for test')


## Execute main function

if __name__ == '__main__':
    main()