"""
preprocess.py

This module handles EEG data loading, normalization, augmentation,
and splitting into training/validation/testing datasets.
"""

import os
import numpy as np
from tqdm import tqdm
import mne
import torch
from torch.utils.data import Dataset, random_split

def print_stats(desc, data, logger=None):
    print(f"{desc} mean: {np.mean(data)}, std: {np.std(data)}, min: {np.min(data)}, max: {np.max(data)}")

def normalize(data, mean_value, std_value, desc="", logger=None):
    data = np.array(data)
    data = (data - mean_value) / std_value
    print_stats(desc, data)
    return list(data)

def filter_freq(data, f_min, f_max, FS):
    return mne.filter.filter_data(np.array(data, dtype=np.float64), FS, f_min, f_max, method="iir", verbose=False)

def downsample(data, FS, FS_new):
    return mne.filter.resample(data, down=FS/FS_new)


def load_data(data_folder, load_labels=True, logger=None):
    fn_list = sorted(os.listdir(data_folder))
    print(f"Loading data from folder: {data_folder} ({len(fn_list)} files) - Load labels {load_labels}")

    data_map = {}
    subject_list = []
    sample_counter = 0

    for fn in tqdm(fn_list):
        if fn.endswith("X.npy"):
            code = fn.split("_")[1][:-4]
        elif fn == "headerInfo.npy":
            meta = np.load(data_folder + fn, allow_pickle=True)
            print(meta)
            continue
        else:
            continue

        eeg = np.load(data_folder + fn, allow_pickle=True)

        if load_labels:
            label_fn = fn.replace("X", "y")
            label = np.load(data_folder + label_fn, allow_pickle=True)
        else:
            label = None

        s_part, r_part = code.split("r")
        subject = int(s_part[1:])
        repetition = int(r_part[:-1])
        
        subject_list.append(subject)

        if subject not in data_map.keys():
            data_map[subject] = {}

        data_map[subject][repetition] = {"eeg": eeg, "label": label}
        sample_counter += len(eeg)

    subject_list = np.unique(subject_list)
    print(f"Loaded total {sample_counter} samples for subjects: {subject_list}")
    return data_map, subject_list

def prepare_window_data(data, subject_list=None, logger=None):
    window_data = []
    window_labels = []

    if subject_list is None:
        subject_list = data.keys()

    for s in tqdm(subject_list):
        for r in data[s].keys():
            eeg = data[s][r]["eeg"]
            label = data[s][r]["label"]

            window_data.extend(eeg)
            
            if label is not None:
                window_labels.extend(label)

    return window_data, window_labels


def get_phase_1_data(config, logger=None):
    """
    Loads phase 1 data from several folders.
    Expects the following keys in config:
      - SOURCE_DATA_FOLDER
      - LEADERBOARD_TARGET_DATA_FOLDER
      - LEADERBOARD_TEST_DATA_FOLDER
      - FINAL_TARGET_DATA_FOLDER
      - FINAL_TEST_DATA_FOLDER
    """
    source_folder = config.get('SOURCE_DATA_FOLDER')
    lb_target_folder = config.get('LEADERBOARD_TARGET_DATA_FOLDER')
    lb_test_folder = config.get('LEADERBOARD_TEST_DATA_FOLDER')
    fn_target_folder = config.get('FINAL_TARGET_DATA_FOLDER')
    fn_test_folder = config.get('FINAL_TEST_DATA_FOLDER')

    # Load source data and compute normalization statistics
    source_data_map, source_subjects = load_data(source_folder, load_labels=True, logger=logger)
    source_data, source_labels = prepare_window_data(source_data_map, source_subjects, logger=logger)
    source_data = np.array(source_data)
    source_mean = np.mean(source_data)
    source_std = np.std(source_data)
    message = f"Source mean: {source_mean}, std: {source_std}, min: {np.min(source_data)}, max: {np.max(source_data)}"
    if logger:
        logger.info(message)
    else:
        print(message)
    source_data = list(source_data)

    # Load and normalize leaderboard target data
    lb_target_data_map, lb_target_subjects = load_data(lb_target_folder, load_labels=True, logger=logger)
    lb_target_data, lb_target_labels = prepare_window_data(lb_target_data_map, lb_target_subjects, logger=logger)
    lb_target_data = normalize(lb_target_data, source_mean, source_std, "Leaderboard target", logger)

    # Load and normalize leaderboard test data (without labels)
    lb_test_data_map, lb_test_subjects = load_data(lb_test_folder, load_labels=False, logger=logger)
    lb_test_data, lb_test_labels = prepare_window_data(lb_test_data_map, lb_test_subjects, logger=logger)
    lb_test_data = normalize(lb_test_data, source_mean, source_std, "Leaderboard test", logger)

    # Load and normalize final target data
    fn_target_data_map, fn_target_subjects = load_data(fn_target_folder, load_labels=True, logger=logger)
    fn_target_data, fn_target_labels = prepare_window_data(fn_target_data_map, fn_target_subjects, logger=logger)
    fn_target_data = normalize(fn_target_data, source_mean, source_std, "Final target", logger)

    # Load and normalize final test data (without labels)
    fn_test_data_map, fn_test_subjects = load_data(fn_test_folder, load_labels=False, logger=logger)
    fn_test_data, fn_test_labels = prepare_window_data(fn_test_data_map, fn_test_subjects, logger=logger)
    fn_test_data = normalize(fn_test_data, source_mean, source_std, "Final test", logger)
    # print("yaha tak sahi chala")
    
    return source_data, source_labels, lb_target_data, lb_target_labels, lb_test_data, lb_test_labels, fn_target_data, fn_target_labels, fn_test_data, fn_test_labels

def get_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, list):
        shape_info = f"(List - First Element Shape: {np.array(data[0]).shape if len(data) > 0 and isinstance(data[0], (list, np.ndarray)) else 'N/A'})"
        return len(data), shape_info
    else:
        return "Unknown Type"

# Dataset class for EEG data
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def preprocess_data(source_data, source_labels, train_split=0.7, val_split=0.15, logger=None):
    """
    Preprocess the EEG data by splitting into train, validation, and test sets.
    Assumes source_data is already in the shape (num_samples, 2, 3000).
    """
    # Ensure source_data and source_labels are NumPy arrays
    source_data = np.array(source_data)
    source_labels = np.array(source_labels)

	# Flatten the source_data to (num_samples, -1)
    source_data = source_data.reshape(source_data.shape[0], -1)
    
    # DO NOT flatten the data; preserve shape (num_samples, 2, 3000)
    source_data = torch.tensor(source_data, dtype=torch.float32)
    source_labels = torch.tensor(source_labels, dtype=torch.long)
    
    dataset = EEGDataset(source_data, source_labels)
    
    # Split into train, validation, and test sets
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    return train_set, val_set, test_set


def preprocess_data_finetune(source_data, source_labels, train_split=0.7, val_split=0.15, logger=None):
    """
    Preprocess the EEG data for fine-tuning (using the same flatten-and-split approach).
    """
    return preprocess_data(source_data, source_labels, train_split, val_split, logger)
