## File to define the function(s) used to control the preprocessing section of the project


## Imports

# From this project (constants)
from utils.constants import SUBJECTS, PATH_DATA_FOLDER, DATA_END_FILE
from utils.constants import XDF_DEJITTER, XDF_SYNC
from utils.constants import STREAM_INFO_NAME, STREAM_TYPE_NAME, STREAM_MARKER_NAME, STREAM_EEG_NAME
from utils.constants import INITIAL_LIST_CH_TO_REMOVE
from utils.constants import STREAM_TIME_SERIES_NAME, STREAM_TIME_STAMPS_NAME, STREAM_DESC_NAME, STREAM_CH_NAME_1, STREAM_CH_NAME_2, STREAM_CH_LABEL_NAME, STREAM_NOMINAL_SR, STREAM_EFFECTIVE_SR
from utils.constants import STREAM_MARKER_VALUE_IDX, STREAM_MARKER_TIME_STAMP_IDX
from utils.constants import TRIGGERS_KEYS
from utils.constants import MNE_STANDARD_MONTAGE, MNE_SR
from utils.constants import MICROVOLTS_TO_VOLTS
from utils.variable_constants import LIST_CHANNELS_TO_REMOVE_BEFOREHAND
from utils.constants import FILTER_FREQ_BANDPASS_ORDER, FILTER_FREQ_NOTCH_ORDER, FILTER_FREQ_NOTCH_CENTER, FILTER_FREQ_NOTCH_HALF_WINDOW
from utils.constants import SPATIAL_FILTER_NONE, SPATIAL_FILTER_CAR
from utils.constants import ANNOTATIONS_ONSET_NAME, ANNOTATIONS_TYPE_NAME
from utils.constants import MI_NAME, REST_NAME, BEGIN_NAME, END_NAME
from utils.constants import MI_LABEL, REST_LABEL
from utils.constants import DO_EPOCH_BASELINE_CORRECTION, EPOCH_BASELINE_CORRECTION_DURATION
from utils.constants import WINDOW_REJECTION_NONE, WINDOW_REJECTION_PTP_150_MICROV
from utils.constants import WINDOW_REJECTION_PTP_150_MICROV_THRESHOLD
from utils.constants import FEATURE_SELECTION_WELCH_GENERAL, FEATURE_SELECTION_CSP_GENERAL, FEATURE_SELECTION_COV_GENERAL
from utils.constants import WELCH_FREQ_BANDS_2, WELCH_FREQ_BANDS_3, WELCH_FREQ_BANDS_SMALL
from utils.variable_constants import RANDOM_SEED

# From this project (functions)
from utils.plots import tmp_plot

# From sys library
import logging

# From external libraries
import mne
import numpy as np
from pyriemann.estimation import Covariances
import pyxdf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch


## Set up logger
logger = logging.getLogger(__name__)

## Set up random behaviour
_rng_preprocessing = np.random.default_rng(RANDOM_SEED)

## Functions
    
# TODO
def read_file_and_create_mne_with_annotations(subject_idx: int, session_idx: int) -> mne.io.RawArray:
    """
    TODO explain

    Parameters:
        subject_idx (int): TODO explain
        session_idx (int): TODO explain

    Returns:
        mne.io.RawArray: TODO explain
    """

    # Load data
    path_file = PATH_DATA_FOLDER + SUBJECTS[subject_idx][session_idx] + DATA_END_FILE
    streams, headers = pyxdf.load_xdf(path_file, dejitter_timestamps=XDF_DEJITTER, synchronize_clocks=XDF_SYNC)
    for stream in streams:
        if stream[STREAM_INFO_NAME][STREAM_TYPE_NAME][0] == STREAM_MARKER_NAME:
            marker_stream = stream
        elif stream[STREAM_INFO_NAME][STREAM_TYPE_NAME][0] == STREAM_EEG_NAME:
            eeg_stream = stream
        else:
            logger.error('Unexpected stream type in file')
            raise ValueError('Unexpected stream type')

    # Get EEG data from stream
    eeg_data = eeg_stream[STREAM_TIME_SERIES_NAME].T
    ch_names = np.sort(np.array([ch[STREAM_CH_LABEL_NAME][0] for ch in eeg_stream[STREAM_INFO_NAME][STREAM_DESC_NAME][0][STREAM_CH_NAME_1][0][STREAM_CH_NAME_2]]))

    # Clean EEG data channels
    idx_chs_to_keep = ~np.isin(ch_names, INITIAL_LIST_CH_TO_REMOVE)
    eeg_data = eeg_data[idx_chs_to_keep, :]
    ch_names = ch_names[idx_chs_to_keep]

    # Get effective SR from time stamps
    eeg_timestamps = eeg_stream[STREAM_TIME_STAMPS_NAME]
    absolute_time_zero = eeg_timestamps[0]
    eeg_timestamps = eeg_timestamps - absolute_time_zero
    effective_sr = np.mean(1 / np.diff(eeg_timestamps))

    # Check SRs
    stream_nominal_sr = float(eeg_stream[STREAM_INFO_NAME][STREAM_NOMINAL_SR][0])
    stream_effective_sr = eeg_stream[STREAM_INFO_NAME][STREAM_EFFECTIVE_SR]
    if (int(effective_sr) != int(stream_nominal_sr)) or (int(effective_sr) != int(stream_effective_sr)) or (int(stream_nominal_sr) != int(stream_effective_sr)):
        logger.warning(f'Mismatch between SRs (rounded to int) [stream nominal - {int(stream_nominal_sr)} Hz || stream effective - {int(stream_effective_sr)} Hz || real effective - {int(effective_sr)} Hz]')

    # Get marker data from stream
    marker_values = np.array([marker[STREAM_MARKER_VALUE_IDX] for marker in marker_stream[STREAM_TIME_SERIES_NAME]])
    marker_timestamps = np.array([marker[STREAM_MARKER_TIME_STAMP_IDX] for marker in marker_stream[STREAM_TIME_SERIES_NAME]])
    marker_timestamps = marker_timestamps - absolute_time_zero

    # Clean marker data
    idx_markers_to_keep = np.isin(marker_values, list(TRIGGERS_KEYS.keys()))
    marker_values = marker_values[idx_markers_to_keep]
    marker_timestamps = marker_timestamps[idx_markers_to_keep]

    # TODO plot markers
    
    # Get channels types and names for MNE
    ch_names = [ch.replace('FP', 'Fp').replace('Z', 'z') for ch in ch_names]
    ch_types = ['eeg'] * len(ch_names)

    # Create MNE object
    info = mne.create_info(ch_names, sfreq=effective_sr, ch_types=ch_types)
    info.set_montage(MNE_STANDARD_MONTAGE)
    eeg_mne = mne.io.RawArray(eeg_data * MICROVOLTS_TO_VOLTS, info)

    # Create and add annotations
    marker_sample_idxs = marker_timestamps * effective_sr
    events = np.zeros(shape=(len(marker_sample_idxs), 3))
    events[:, 0] = marker_sample_idxs.astype(int)
    events[:, 2] = marker_values.astype(int)
    annotations = mne.annotations_from_events(events=events, sfreq=effective_sr, event_desc=TRIGGERS_KEYS)
    eeg_mne.set_annotations(annotations)
    logger.info(f'Annotations: {eeg_mne.annotations.count()}')

    return eeg_mne

# TODO
def resample_and_filter_mne_frequency_and_spatially(eeg_mne: mne.io.RawArray, resample_sr: int, low_freq_bandpass: float, high_freq_bandpass: float, type_spatial_filter: str, remove_some_channels_beforehand: bool) -> mne.io.RawArray:
    """
    TODO explain (right now only butter)

    Parameters:
        eeg_mne (mne.io.RawArray): TODO explain
        resample_sr (int): TODO explain
        low_freq_bandpass (float): TODO explain
        high_freq_bandpass (float): TODO explain
        type_spatial_filter (str): TODO explain
        remove_some_channels_beforehand (bool): TODO explains

    Returns:
        mne.io.RawArray: TODO explain
    """

    # Create copy to avoid modyfing original variable
    filtered_eeg_mne = eeg_mne.copy()

    # Resample to ensure same SR
    filtered_eeg_mne.resample(resample_sr)
    logger.info(f'Data resampled to {resample_sr} Hz')
    logger.warning('Do we want to resample here or after epochs/windows? (do we care about resample for rejection?)')

    # If necessary remove selected channels
    if remove_some_channels_beforehand:
        # Ensure all channels exist
        existing_channels = list(filtered_eeg_mne.ch_names) 
        missing_channels = [ch for ch in LIST_CHANNELS_TO_REMOVE_BEFOREHAND if ch not in existing_channels]
        if len(missing_channels) != 0:
            logger.error(f'Unexpected channel selection, the following channels are not present in data: {missing_channels}')
            raise ValueError('Unexpected channel selection')
        filtered_eeg_mne.drop_channels(LIST_CHANNELS_TO_REMOVE_BEFOREHAND)
        logger.info(f'List of channels removed beforehand: {LIST_CHANNELS_TO_REMOVE_BEFOREHAND}')
    else:
        logger.info(f'0 channels removed before filtering')

    # Filter in frequency (bandpass)
    filtered_eeg_mne.filter(l_freq=low_freq_bandpass, h_freq=high_freq_bandpass, method='iir', iir_params=dict(order=FILTER_FREQ_BANDPASS_ORDER, ftype='butter'))
    logger.info(f'Bandpass filter [{low_freq_bandpass}, {high_freq_bandpass}] Hz')

    # Filter in frequency (notch) TODO cascade??
    notch_low = FILTER_FREQ_NOTCH_CENTER + FILTER_FREQ_NOTCH_HALF_WINDOW
    notch_high = FILTER_FREQ_NOTCH_CENTER - FILTER_FREQ_NOTCH_HALF_WINDOW
    filtered_eeg_mne.filter(l_freq=notch_low, h_freq=notch_high, method='iir', iir_params=dict(order=FILTER_FREQ_NOTCH_ORDER, ftype='butter'))
    logger.info(f'Notch filter [{notch_high}, {notch_low}] Hz')

    # Spatial filter
    logger.info(f'Type of spatial filter used: {type_spatial_filter}')
    if type_spatial_filter == SPATIAL_FILTER_NONE:
        pass
    elif type_spatial_filter == SPATIAL_FILTER_CAR:
        filtered_eeg_mne.set_eeg_reference('average')
    else:
        logger.error(f'Unexpected spatial filter option')
        raise ValueError('Unexpected spatial filter option')

    logger.warning('Code small lap and big lap, we have to determine neighbours ugh')

    return filtered_eeg_mne

# TODO
def characterization_and_channel_selection(a, b):
    return None

# TODO
def filter_channels_mne(eeg_mne: mne.io.RawArray, list_channels_to_use: list[str]) -> mne.io.RawArray:
    """
    TODO explain

    Parameters:
        eeg_mne (mne.io.RawArray): TODO explain
        list_channels_to_use (list[str]): TODO explain

    Returns:
        mne.io.RawArray: TODO explain
    """
    
    # Create copy to avoid modyfing original variable
    filtered_eeg_mne = eeg_mne.copy()

    # Ensure all channels exist
    existing_channels = list(filtered_eeg_mne.ch_names) 
    missing_channels = [ch for ch in list_channels_to_use if ch not in existing_channels]
    if len(missing_channels) != 0:
        logger.error(f'Unexpected channel selection, the following channels are not present in data: {missing_channels}')
        raise ValueError('Unexpected channel selection')

    # Filter channels
    filtered_eeg_mne = filtered_eeg_mne.pick_channels(list_channels_to_use)
    logger.info(f'Final list of channels in data: {list_channels_to_use}')
        
    return filtered_eeg_mne

# TODO
def epoch_mne(eeg_mne: mne.io.RawArray, epoch_rejection_criteria: str) -> tuple[list[np.ndarray], list[int]]:
    """
    TODO explain

    Parameters:
        eeg_mne (mne.io.RawArray): TODO explain
        epoch_rejection_criteria (str): TODO explain

    Returns:
        list[np.ndarray]: TODO explain
        list[int]: TODO
    """
    
    # Get list of epochs
    this_sr = eeg_mne.info[MNE_SR]
    last_type = None
    epoch_labels = []
    epoch_starts = []
    epoch_ends = []
    for annotation in eeg_mne.annotations:
        # Read this annotation
        this_type = annotation[ANNOTATIONS_TYPE_NAME]
        this_time = annotation[ANNOTATIONS_ONSET_NAME]
        
        # Get label
        if MI_NAME in this_type:
            this_label = MI_LABEL
        elif REST_NAME in this_type:
            this_label = REST_LABEL
        else:
            logger.error('Unexpected annotation, should be a MI or REST one')
            raise ValueError('Unexpected annotation')

        # First annotation (has to be begin)
        if last_type == None:
            if BEGIN_NAME in this_type:
                last_type = this_type
                epoch_labels.append(this_label)
                epoch_starts.append(this_time)
            else:
                logger.error('Unexpected first annotation, should be a BEGIN one')
                raise ValueError('Unexpected annotation')
        # Last annotation was a begin xx, now we should have a end xx
        elif BEGIN_NAME in last_type:
            if (END_NAME in this_type) and (((MI_NAME in last_type) and (MI_NAME in this_type)) or ((REST_NAME in last_type) and (REST_NAME in this_type))):
                last_type = this_type
                epoch_ends.append(this_time)
            else:
                logger.error('Unexpected annotation, should be a END of the same type as last one')
                raise ValueError('Unexpected annotation')
        # Last annotation was a end xx, now we should have a begin whatever
        elif END_NAME in last_type:
            if BEGIN_NAME in this_type:
                last_type = this_type
                epoch_labels.append(this_label)
                epoch_starts.append(this_time)
            else:
                logger.error('Unexpected annotation, should be a BEGIN one')
                raise ValueError('Unexpected annotation')
        # Unexpected annotation
        else:
            logger.error('Unexpected annotation, should be a BEGIN or END one')
            raise ValueError('Unexpected annotation')

    # Check last annotation was a END one
    if END_NAME not in last_type:
        logger.error('Unexpected last annotation, should be an END one')
        raise ValueError('Unexpected annotation')

    # TODO reject epochs
    logger.warning('We currently dont do epoch rejection')

    if DO_EPOCH_BASELINE_CORRECTION:
        logger.info(f'Doing an epoch baseline correction of {EPOCH_BASELINE_CORRECTION_DURATION} s')
        epoch_baseline_correction_duration_samples = int(round(EPOCH_BASELINE_CORRECTION_DURATION * this_sr))
    else:
        logger.info('No epoch baseline correction applied')

    # Crop EEG data into epochs 
    epoch_starts_sample = [int(round(start * this_sr)) for start in epoch_starts]
    epoch_end_sample = [int(round(end * this_sr)) for end in epoch_ends]
    list_epochs_np = []
    for i in range(len(epoch_labels)):
        this_epoch = eeg_mne.get_data(start=epoch_starts_sample[i], stop=epoch_end_sample[i])
        if DO_EPOCH_BASELINE_CORRECTION:
            this_baseline = eeg_mne.get_data(start=(epoch_starts_sample[i] - epoch_baseline_correction_duration_samples), stop=epoch_starts_sample[i])
            this_epoch -= np.mean(this_baseline, axis=1, keepdims=True)
        list_epochs_np.append(this_epoch)

    logger.info(f'Number of epochs: [MI - {epoch_labels.count(MI_LABEL)} || REST - {epoch_labels.count(REST_LABEL)}]')

    return (list_epochs_np, epoch_labels)

# TODO
def window_epochs_np(list_epochs_np: list[np.ndarray], epoch_labels: list[int], sr: float, duration_window: float, dt_window: float, window_rejection_criteria: str) -> tuple[list[np.ndarray], list[int], list[int]]:
    """
    TODO explain

    Parameters:
        list_epochs_np (list[np.ndarray]): TODO explain
        epoch_labels (list[int]): TODO explain
        sr (float): TODO explain
        duration_window (float): TODO explain
        dt_window (float): TODO explain
        epoch_rejection_criteria (str): TODO explain

    Returns:
        list[np.ndarray]: TODO explain
        list[int]: TODO
        list[int]: TODO
    """

    # Get samples of windows
    duration_window_samples = int(round(duration_window * sr))
    dt_window_samples = int(round(dt_window * sr))

    # Get list of windows by iterating each epoch
    list_windows_np = []
    windows_labels = []
    windows_epochs = []
    rejected_window_mi = 0
    rejected_window_rest = 0
    for epoch_idx, epoch in enumerate(list_epochs_np):
        n_samples_epoch = epoch.shape[1]
        this_start = 0 - dt_window_samples
        this_end = this_start + duration_window_samples
        this_label = epoch_labels[epoch_idx]
        this_epoch = epoch_idx
        while (this_end + dt_window_samples) < n_samples_epoch:
            this_start = this_start + dt_window_samples
            this_end = this_end + dt_window_samples
            this_window = epoch[:, this_start:this_end]
            add_this = True

            # Reject
            if window_rejection_criteria == WINDOW_REJECTION_NONE:
                pass
            elif window_rejection_criteria == WINDOW_REJECTION_PTP_150_MICROV:
                if np.any(np.ptp(this_window, axis=1) > WINDOW_REJECTION_PTP_150_MICROV_THRESHOLD):
                    add_this = False
            else:
                logger.error(f'Unexpected window rejection option')
                raise ValueError('Unexpected window rejection option')

            if add_this:
                list_windows_np.append(this_window)
                windows_labels.append(this_label)
                windows_epochs.append(this_epoch)
            else:
                if this_label == MI_LABEL:
                    rejected_window_mi += 1
                elif this_label == REST_LABEL:
                    rejected_window_rest += 1
                else:
                    logger.error(f'Unexpected label value')
                    raise ValueError('Unexpected label value')

    total_rejected_window = rejected_window_mi + rejected_window_rest

    logger.info(f'Windows created of shape {list_windows_np[0].shape}')
    logger.info(f'Rejection criteria: {window_rejection_criteria}')
    logger.info(f'Rejected windows: {total_rejected_window} (MI - {rejected_window_mi} || REST - {rejected_window_rest}) / {total_rejected_window + len(list_windows_np)}')
    logger.info(f'Number of windows: [MI - {windows_labels.count(MI_LABEL)} || REST - {windows_labels.count(REST_LABEL)}]')

    return list_windows_np, windows_labels, windows_epochs

# TODO
def create_folds_np(list_windows_np: list[np.ndarray], windows_labels: list[int], windows_epochs: list[int]) -> list[tuple[list[np.ndarray], list[np.ndarray], list[int], list[int], list[int], list[int]]]:
    """
    TODO explain

    Parameters:
        list_windows_np (list[np.ndarray]): TODO explain
        windows_labels (list[int]): TODO explain
        windows_labels (list[int]): TODO explain

    Returns:
        list[tuple[list[np.ndarray], list[np.ndarray], list[int], list[int], list[int], list[int]]: TODO explain
    """

    # Get epoch IDs for MI and REST
    unique_epoch_ids = np.unique(windows_epochs)
    mi_epoch_ids = []
    rest_epoch_ids = []
    for epoch_id in unique_epoch_ids:
        this_label = windows_labels[windows_epochs.index(epoch_id)]
        if this_label == MI_LABEL:
            mi_epoch_ids.append(epoch_id)
        elif this_label == REST_LABEL:
            rest_epoch_ids.append(epoch_id)
        else:
            logger.error('Unexpected label, should be a MI or REST one')
            raise ValueError('Unexpected annotation')

    logger.warning('while doing LOOOCV im asuming equal number of epochs for MI and REST, thats ok?')
    # Make sure there is an equal number of epochs per label
    if len(mi_epoch_ids) != len(rest_epoch_ids):
        logger.error('Number of epochs of MI and REST in train should be the same')
        raise ValueError('Unexpected number of epochs per class')

    # Create folds for each pair of MI and REST epochs
    list_folds = []
    np_list_windows_np = np.stack(list_windows_np, axis=0)
    np_windows_labels = np.array(windows_labels)
    np_windows_epochs = np.array(windows_epochs)
    for idx, this_mi_epoch in enumerate(mi_epoch_ids):
        this_rest_epoch = rest_epoch_ids[idx]

        # Get idxs of windows for validation
        idxs_val_mi = np.where(np_windows_epochs == this_mi_epoch)[0]
        idxs_val_rest = np.where(np_windows_epochs == this_rest_epoch)[0]
        idxs_val = np.concatenate([idxs_val_mi, idxs_val_rest])

        # Get idxs of windows for training
        all_idxs = np.arange(len(windows_epochs))
        idxs_train = np.setdiff1d(all_idxs, idxs_val)

        # Get the samples and labels of these idxs
        train_fold = np_list_windows_np[idxs_train]
        val_fold = np_list_windows_np[idxs_val]
        train_labels = np_windows_labels[idxs_train]
        val_labels = np_windows_labels[idxs_val]
        train_epochs = np_windows_epochs[idxs_train]
        val_epochs = np_windows_epochs[idxs_val]

        # Convert back to lists
        train_fold = list(train_fold)
        val_fold = list(val_fold)
        train_labels = list(train_labels)
        val_labels = list(val_labels)
        train_epochs = list(train_epochs)
        val_epochs = list(val_epochs)

        list_folds.append((train_fold, val_fold, train_labels, val_labels, train_epochs, val_epochs))

    return list_folds

# TODO
def feature_selection(train_list_windows_np: list[np.ndarray], test_list_windows_np: list[np.ndarray], feature_selection_option: str, sr: float, train_labels: list[int]):
    """
    TODO explain

    Parameters:
        list_windows_np (list[np.ndarray]): TODO explain
        feature_selection_option (str): TODO explain
        sr (float): TODO

    Returns:
        TODO: TODO explain
    """

    # Apply the selected feature selection to train

    # Get avg band power per frequency band per channel per window
    if FEATURE_SELECTION_WELCH_GENERAL in feature_selection_option:
        # Get details for welch
        freq_bands_to_use_and_bool_pca = feature_selection_option.split('welch_')[1]
        freq_bands_to_use = freq_bands_to_use_and_bool_pca.split('_')[0]
        bool_pca = False if 'no' in freq_bands_to_use_and_bool_pca.split('_freq_bands_')[1] else True
        
        if bool_pca == True:
            pca_components = int(freq_bands_to_use_and_bool_pca.split('_freq_bands_')[1].split('_')[1])

        if freq_bands_to_use == '2':
            welch_freq_bands = WELCH_FREQ_BANDS_2
        elif freq_bands_to_use == '3':
            welch_freq_bands = WELCH_FREQ_BANDS_3
        elif freq_bands_to_use == 'small':
            welch_freq_bands = WELCH_FREQ_BANDS_SMALL
        else:
            logger.error(f'Unexpected welch freq bands option')
            raise ValueError('Unexpected welch freq bands option')

        # Get train
        train_features = []
        for window in train_list_windows_np:
            this_window_features = []
            for ch_idx in range(window.shape[0]):
                freqs, power = welch(window[ch_idx, :], fs=sr, nperseg=min(256, window.shape[1]))
                for min_freq, max_freq in welch_freq_bands:
                    this_bandpower = power[(freqs >= min_freq) & (freqs < max_freq)].mean()
                    this_window_features.append(this_bandpower)
            train_features.append(this_window_features)
        train_features = np.array(train_features)

        # Get scaler and scale train
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        # If necessary, train PCA and apply to train
        if bool_pca:
            pca = PCA(n_components=pca_components)
            train_features = pca.fit_transform(train_features)

        # After fitting scaler, do test
        test_features = []
        for window in test_list_windows_np:
            this_window_features = []
            for ch_idx in range(window.shape[0]):
                freqs, power = welch(window[ch_idx, :], fs=sr, nperseg=min(256, window.shape[1]))
                for min_freq, max_freq in welch_freq_bands:
                    this_bandpower = power[(freqs >= min_freq) & (freqs < max_freq)].mean()
                    this_window_features.append(this_bandpower)
            test_features.append(this_window_features)
        test_features = np.array(test_features)
        test_features = scaler.transform(test_features)

        # If necessary, apply PCA to test
        if bool_pca:
            test_features = pca.transform(test_features)

    # Get CSP features
    elif FEATURE_SELECTION_CSP_GENERAL in feature_selection_option:
        # Get details for CSP
        n_components_and_bool_log = feature_selection_option.split('csp_')[1]
        n_components = int(n_components_and_bool_log.split('_')[0])
        bool_log = False if 'no' in n_components_and_bool_log.split('_')[1] else True

        # Create CSP fitter
        csp = mne.decoding.CSP(n_components=n_components, log=bool_log)

        # Fit CSP and get train
        train_features = csp.fit_transform(np.stack(train_list_windows_np), np.array(train_labels))

        # Get scaler and scale train
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        # After fitting CSP and scaler, do test
        test_features = csp.transform(np.stack(test_list_windows_np))
        test_features = scaler.transform(test_features)

    # Get cov features
    elif FEATURE_SELECTION_COV_GENERAL in feature_selection_option:
        # Get details for cov (right now I don't use this)
        cov_info = feature_selection_option.split('cov_')[1]

        # Create cov fitter
        cov = Covariances(estimator='oas')

        # Fit cov and get train
        train_features = cov.fit_transform(np.stack(train_list_windows_np))

        # For COV + MDM we shouldn't use standar scaler, right?

        # Get test
        test_features = cov.transform(np.stack(test_list_windows_np))

    # Report if there is an unexpected feature selections
    else:
        logger.error(f'Unexpected feature selection option')
        raise ValueError('Unexpected feature selection option')

    return train_features, test_features

# TODO
def shuffle_samples(features: np.ndarray, labels: list[int], epochs: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO explain

    Parameters:
        features (np.ndarray): TODO explain
        labels (list[int]): TODO explain
        epochs (list[int]): TODO explains

    Returns:
        tuple[np.ndarray, np.ndarray]: TODO explain
    """

    # Get random permutation of idxs
    random_idxs = _rng_preprocessing.permutation(len(labels))

    # Shuffle
    random_order_features = features[random_idxs]
    random_order_labels = np.array(labels)[random_idxs]
    random_order_epochs = np.array(epochs)[random_idxs]

    return (random_order_features, random_order_labels, random_order_epochs)