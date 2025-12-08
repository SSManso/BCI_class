## File to define the main function of the project


## Imports

# From this project (constants)
from utils.variable_constants import LOGGER_LEVEL_CONSOLE, LOGGER_LEVEL_FILE, LOG_FILE_NAME, LIBRARIES_TO_IGNORE_LOGS_FROM
from utils.variable_constants import TRAIN_SUBJECT_IDX_TO_USE, TRAIN_SESSION_IDX_TO_USE, TEST_SUBJECT_IDX_TO_USE, TEST_SESSION_IDX_TO_USE
from utils.variable_constants import EXTRA_TRAIN_SUBJECT_IDX_TO_USE, EXTRA_TRAIN_SESSION_IDX_TO_USE
from utils.variable_constants import RESAMPLE_SR, LOW_FREQ_BANDPASS, HIGH_FREQ_BANDPASS, TYPE_SPATIAL_FILTER, REMOVE_SOME_CHANNELS_BEFOREHAND
from utils.variable_constants import EPOCH_REJECTION_CRITERIA
from utils.variable_constants import WINDOW_REJECTION_CRITERIA
from utils.variable_constants import CH_SELECTION
from utils.variable_constants import DURATION_WINDOW, DT_WINDOW
from utils.variable_constants import DO_VALIDATION, NO_VALIDATION_FEATURE, NO_VALIDATION_MODEL
from utils.variable_constants import N_VALIDATION_FOLDS
from utils.variable_constants import LIST_FEATURE_SELECTION_OPTIONS, LIST_MODEL_OPTIONS
from utils.constants import METRIC_ACC
from utils.constants import N_DECIMALS
from utils.variable_constants import RANDOM_SEED
from utils.constants import N_ITERATIONS_RANDOM_CHANCE
from utils.constants import SUBJECTS_NAME, SESSIONS_NAME

# From this project (functions)
from utils.create_logger import create_logger
from utils.preprocessing import read_file_and_create_mne_with_annotations
from utils.preprocessing import resample_and_filter_mne_frequency_and_spatially
from utils.preprocessing import characterization_and_channel_selection
from utils.preprocessing import filter_channels_mne
from utils.preprocessing import epoch_mne
from utils.preprocessing import window_epochs_np
from utils.preprocessing import create_folds_np
from utils.preprocessing import feature_selection
from utils.preprocessing import shuffle_samples
from utils.decoder import create_model
from utils.decoder import train_model
from utils.decoder import test_model
from utils.decoder import average_list_metrics
from utils.decoder import max_list_metrics
from utils.decoder import log_metrics
from utils.decoder import compute_epoch_level_metrics
from utils.decoder import plot_confusion_matrix
from utils.decoder import compute_epoch_level_predictions

# From sys library
from datetime import datetime
import logging
import time

# From external libraries
import numpy as np

## Set up random behaviour
_rng_main = np.random.default_rng(RANDOM_SEED)


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
    list_channels_to_use = CH_SELECTION
    logger.info(f'Chosen channels: {list_channels_to_use}')
    train_clean_ch_filt_mne = filter_channels_mne(train_filt_mne, list_channels_to_use)
    test_clean_ch_filt_mne = filter_channels_mne(test_filt_mne, list_channels_to_use)

    # Epoch data (full trial)
    logger.info('-- Train -- Epoch data')
    train_epoch_np, train_epoch_labels = epoch_mne(train_clean_ch_filt_mne, EPOCH_REJECTION_CRITERIA)
    logger.info('-- Test -- Epoch data')
    test_epoch_np, test_epoch_labels = epoch_mne(test_clean_ch_filt_mne, EPOCH_REJECTION_CRITERIA)

    # Create samples for model (small windows)
    logger.info('-- Train -- Windowing epochs')
    train_windows_np, final_train_labels, final_train_epochs = window_epochs_np(train_epoch_np, train_epoch_labels, RESAMPLE_SR, DURATION_WINDOW, DT_WINDOW, WINDOW_REJECTION_CRITERIA)
    logger.info('-- Test -- Windowing epochs')
    test_windows_np, final_test_labels, final_test_epochs = window_epochs_np(test_epoch_np, test_epoch_labels, RESAMPLE_SR, DURATION_WINDOW, DT_WINDOW, WINDOW_REJECTION_CRITERIA)

    logger.warning('Show the std when you do avg of metrics not only avg, both in validation and random chance')
    logger.warning('Look at doing probabilities not only prediction of labels')
    logger.warning('Right now metric is just per-window accuracy on all samples')
    logger.warning('Right now Im just doing LOOCV, change so that you can choose between LOOCV and LO-run-OCV')

    if DO_VALIDATION:
        # Grid search + validation (k-fold) for best feature selection + model (ensure windows from the same trial fall in the same fold!)
        logger.info('-- Validation -- Create folds')
        list_folds_train_windows_np = create_folds_np(train_windows_np, final_train_labels, final_train_epochs)
        all_combinations = [(each_feature_selection_option, each_model_option) for each_feature_selection_option in LIST_FEATURE_SELECTION_OPTIONS for each_model_option in LIST_MODEL_OPTIONS]
        clean_combinations = []
        for this_feature, this_model in all_combinations:
            this_feature_group = this_feature.split('group_')[1].split('_')[0]
            this_model_group = this_model.split('group_')[1].split('_')[0]
            if this_feature_group == this_model_group:
                clean_combinations.append((this_feature, this_model))
        logger.info(f'All possible comnbinations ({len(clean_combinations)}): {clean_combinations}')
        
        validation_metric = []
            # - Grid search loop
        for this_feature_selection_option, this_model_option in clean_combinations:
            this_combination_metric = []

                # - k-fold loop
            logger.info(f'-- Validation -- k-fool loop for combination [feature - {this_feature_selection_option} || model - {this_model_option}]')
            t_0 = time.time()
            for train_fold_np, val_fold_np, train_labels, val_labels, train_epochs, val_epochs in list_folds_train_windows_np:

                # Feature selection
                train_input_model, val_input_model = feature_selection(train_fold_np, val_fold_np, this_feature_selection_option, RESAMPLE_SR, train_labels)

                # Shuffle samples
                train_input_model, train_labels, train_epochs = shuffle_samples(train_input_model, train_labels, train_epochs)
                val_input_model, val_labels, val_epochs = shuffle_samples(val_input_model, val_labels, val_epochs)

               # Train and validate model
                raw_model = create_model(this_model_option)
                trained_model, train_metric = train_model(raw_model, train_input_model, train_labels)
                val_metric = test_model(trained_model, val_input_model, val_labels)
                val_epoch_metrics = compute_epoch_level_metrics(trained_model, val_input_model, val_labels, val_epochs, prefix="epoch_")
                val_metric.update(val_epoch_metrics)
                this_combination_metric.append(val_metric)
            
            # Save result
            validation_metric.append(average_list_metrics(this_combination_metric))
            # logger.info(f'Val performance: {validation_metric[-1][METRIC_ACC]:.{N_DECIMALS}f}')
            log_metrics(logger, "Validation", validation_metric[-1], N_DECIMALS)

            t_f = time.time()
            logger.info(f'Completed in {int(round(t_f - t_0))} seconds')

        final_feature_selection_option, final_model_option = clean_combinations[validation_metric.index(max_list_metrics(validation_metric))]
    else:
        logger.info('-- Validation -- Skipped')
        final_feature_selection_option, final_model_option = NO_VALIDATION_FEATURE, NO_VALIDATION_MODEL

    # If needed do one extra train file
    if (EXTRA_TRAIN_SUBJECT_IDX_TO_USE is None) or (EXTRA_TRAIN_SESSION_IDX_TO_USE is None):
        logger.info('-- Extra Train -- Skip')
    else:
        # Get extra train
        logger.info('-- Extra Train -- Read data')
        extra_train_raw_mne = read_file_and_create_mne_with_annotations(EXTRA_TRAIN_SUBJECT_IDX_TO_USE, EXTRA_TRAIN_SESSION_IDX_TO_USE)
        logger.info('-- Extra Train -- Filter data')
        extra_train_filt_mne = resample_and_filter_mne_frequency_and_spatially(extra_train_raw_mne, RESAMPLE_SR, LOW_FREQ_BANDPASS, HIGH_FREQ_BANDPASS, TYPE_SPATIAL_FILTER, REMOVE_SOME_CHANNELS_BEFOREHAND)
        extra_train_clean_ch_filt_mne = filter_channels_mne(extra_train_filt_mne, list_channels_to_use)
        logger.info('-- Extra Train -- Epoch data')
        extra_train_epoch_np, extra_train_epoch_labels = epoch_mne(extra_train_clean_ch_filt_mne, EPOCH_REJECTION_CRITERIA)
        logger.info('-- Extra Train -- Windowing epochs')
        extra_train_windows_np, extra_final_train_labels, extra_final_train_epochs = window_epochs_np(extra_train_epoch_np, extra_train_epoch_labels, RESAMPLE_SR, DURATION_WINDOW, DT_WINDOW, WINDOW_REJECTION_CRITERIA)
    
        # Make epoch IDs unique
        epoch_id_offset = np.max(final_train_epochs) + 1
        extra_final_train_epochs = [epoch_label + epoch_id_offset for epoch_label in extra_final_train_epochs]

        # Merge both trains
        train_windows_np = train_windows_np + extra_train_windows_np
        final_train_labels = final_train_labels + extra_final_train_labels
        final_train_epochs = final_train_epochs + extra_final_train_epochs

    # Test final model
    logger.info('-- Test -- Evaluate final model')
    t_0 = time.time()
        # - Feature selection
    train_input_model, test_input_model = feature_selection(train_windows_np, test_windows_np, final_feature_selection_option, RESAMPLE_SR, final_train_labels)
        # - Shuffle samples
    train_input_model, final_train_labels, final_train_epochs = shuffle_samples(train_input_model, final_train_labels, final_train_epochs)
    test_input_model, final_test_labels, final_test_epochs = shuffle_samples(test_input_model, final_test_labels, final_test_epochs)
        # - Train and test model
    raw_model = create_model(final_model_option)
    trained_model, train_metric = train_model(raw_model, train_input_model, final_train_labels)
    final_test_metric = test_model(trained_model, test_input_model, final_test_labels)
    test_epoch_metrics = compute_epoch_level_metrics(
        trained_model, test_input_model, final_test_labels, final_test_epochs, prefix="epoch_",)
    final_test_metric.update(test_epoch_metrics)

    plot_confusion_matrix(
        true=final_test_labels,
        pred=trained_model.predict(test_input_model),
        class_names=("REST", "MI"),
        title="Window-Level Confusion Matrix (Final Test)",
        normalize=False,
        save_path=f"./plots/{LOG_FILE_NAME}_cm_window.png",
    )

        # - Get random chance level
    all_random_metrics = []
    for _ in range(N_ITERATIONS_RANDOM_CHANCE):
        random_idxs = _rng_main.permutation(len(final_test_labels))
        random_test_labels = final_test_labels[random_idxs]
        random_test_epochs = final_test_epochs[random_idxs]
        random_metric = test_model(trained_model, test_input_model, random_test_labels)
        random_epoch_metrics = compute_epoch_level_metrics(
            trained_model, test_input_model, random_test_labels, random_test_epochs, prefix="epoch_",)
        random_metric.update(random_epoch_metrics)
        all_random_metrics.append(random_metric)
    random_chance_metric = average_list_metrics(all_random_metrics)

    # --- Epoch-level confusion matrix ---
    test_epoch_true, test_epoch_pred = compute_epoch_level_predictions(trained_model, test_input_model, final_test_labels, final_test_epochs)

    plot_confusion_matrix(true=test_epoch_true,pred=test_epoch_pred, class_names=("REST", "MI"), title="Epoch-Level Confusion Matrix (Final Test)", normalize=False, save_path=f"./plots/{LOG_FILE_NAME}_cm_epoch.png")

    
    t_f = time.time()
    logger.info(f'Completed in {int(round(t_f - t_0))} seconds')

    logger.info('-- Final results --')
    logger.info(f'Train session: {SUBJECTS_NAME[TRAIN_SUBJECT_IDX_TO_USE]} - {SESSIONS_NAME[TRAIN_SESSION_IDX_TO_USE]}')
    logger.info(f'Test session: {SUBJECTS_NAME[TEST_SUBJECT_IDX_TO_USE]} - {SESSIONS_NAME[TEST_SESSION_IDX_TO_USE]}')
    logger.info(f'Feature selection type: {final_feature_selection_option}')
    logger.info(f'Model type: {final_model_option}')
    # logger.info(f'Final performance: {final_test_metric[METRIC_ACC]:.{N_DECIMALS}f}')
    # logger.info(f'Random chance level: {random_chance_metric[METRIC_ACC]:.{N_DECIMALS}f}')

    log_metrics(logger, "Final Test", final_test_metric, N_DECIMALS)
    log_metrics(logger, "Random chance level", final_test_metric, N_DECIMALS)

    logger.warning('feature importance heatmap, kinda like shapley?')


## Execute main function

if __name__ == '__main__':
    main()