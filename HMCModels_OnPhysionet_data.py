"""
main.py

This is the main runner script for cross-dataset evaluation.
It sets up logging (with logfile "cross_dataset.log"),
creates a fixed evaluation output directory ("modelOnHMC-testOnPhysionet"),
loads and preprocesses the HMC EEG dataset along with LB and FN data splits,
and then iterates through model folders (from the specified outputs directory).
For each folder, it extracts model details from the folder name,
loads the model (if a valid "models/whisper_best.pt" file exists),
adjusts the final classifier if there is a class-dimension mismatch,
and evaluates it on:
  - The source test set,
  - LB (65–80 age group) target test set,
  - LB actual test set,
  - FN (80+ age group) target test set,
  - FN actual test set.
Evaluation results and diagrams are saved in subfolders.
Folders missing the model file or not following the expected naming format are skipped.
"""

import os
import sys
import logging
import pickle
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

# Import your custom functions and classes.
from preprocessing.preprocess import get_phase_1_data, preprocess_data, preprocess_data_finetune
from model.model_builder import get_model
from trainandeval.eval import evaluate, preprocess_whisper_input

def setup_logging(output_dir, log_filename="cross_dataset.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = os.path.join(output_dir, log_filename)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(labels, preds, num_classes, save_path):
    """
    Computes and plots the confusion matrix.
    
    Args:
      labels (array-like): True labels.
      preds (array-like): Predicted labels.
      num_classes (int): Number of classes.
      save_path (str): File path to save the confusion matrix image.
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_with_confusion_matrix(model, loader, device, preprocess_fn=None, num_classes=6, save_dir='outputs'):
    """
    Wrapper for the evaluate function that adds confusion matrix saving logic.
    
    Calls your original evaluate function and then saves a confusion matrix image.
    
    Returns the same outputs as evaluate.
    """
    # Call your original evaluate function (which remains unchanged)
    metrics, all_labels, all_preds, all_probs = evaluate(model, loader, device, preprocess_fn, num_classes, save_dir)
    
    # Define path for confusion matrix image.
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    
    # Plot and save the confusion matrix.
    plot_confusion_matrix(all_labels, all_preds, num_classes, cm_path)
    
    return metrics, all_labels, all_preds, all_probs




def main():
    # Fixed configuration for evaluation.
    fixed_config = {
        "batch_size": 16,
        "num_classes": 6,       # Our evaluation dataset has 6 classes.
        "input_channels": 2,
		"DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/",
		"SOURCE_DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/SleepSource/",
		"LEADERBOARD_TARGET_DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/LeaderboardSleep/sleep_target/",
		"LEADERBOARD_TEST_DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/LeaderboardSleep/testing/",
		"FINAL_TARGET_DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/finalSleep/sleep_target/",
		"FINAL_TEST_DATA_FOLDER": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/finalSleep/testing/",
		"LB_TEST_LABELS_FILE": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/LeaderboardSleep/leaderboardOriginal.npy",
		"FN_TEST_LABELS_FILE": "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/Sleep_Data/finalSleep/FinalOriginal.npy"
    }

    # Set seeds for reproducibility so that dataset splits remain consistent.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Instead of loading a YAML file, we define the config inline.
    config = fixed_config.copy()
    # (Optionally add any additional config keys needed by get_phase_1_data here.)

    # Set evaluation output directory name.
    eval_output_dir = "modelOnHMC-testOnPhysionet"
    os.makedirs(eval_output_dir, exist_ok=True)

    logger = setup_logging(eval_output_dir)
    logger.info("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Starting experiment with config: %s", config)
    logger.info("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    # Load phase 1 data from all sources.
    # Expected to return:
    # source_data, source_labels,
    # lb_target_data, lb_target_labels, lb_test_data, lb_test_labels,
    # fn_target_data, fn_target_labels, fn_test_data, fn_test_labels
    (source_data, source_labels, 
     lb_target_data, lb_target_labels, lb_test_data, lb_test_labels, 
     fn_target_data, fn_target_labels, fn_test_data, fn_test_labels) = get_phase_1_data(config, logger)

    # Optionally load test labels from files if specified in config.
    if "LB_TEST_LABELS_FILE" in config:
        lb_test_labels = np.load(config["LB_TEST_LABELS_FILE"])
    if "FN_TEST_LABELS_FILE" in config:
        fn_test_labels = np.load(config["FN_TEST_LABELS_FILE"])

    # Preprocess the source data (we only use the test set for evaluation).
    train_set, val_set, test_set = preprocess_data(source_data, source_labels, logger=logger)
    logger.info("Source data: Train set size: %d, Val set size: %d, Test set size: %d",
                len(train_set), len(val_set), len(test_set))
    test_loader = DataLoader(test_set, batch_size=fixed_config.get("batch_size", 16), shuffle=False)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Define the directory where model outputs are stored.
    models_parent_dir = "/home/deepak/Documents/Deepak/Students/Susmit_23CS60R75/wisper/whisper_code_HMC_data/outputs"
    if not os.path.exists(models_parent_dir):
        logger.error("Models parent directory %s does not exist.", models_parent_dir)
        sys.exit(1)

    # Dictionary to store evaluation metrics for each model folder.
    all_eval_metrics = {}

    # Iterate over each folder in models_parent_dir.
    for folder in os.listdir(models_parent_dir):
        folder_path = os.path.join(models_parent_dir, folder)
        if os.path.isdir(folder_path):
            # Expected folder name format: "Whisper_<variant>_<pretrained/without_pretrain>_..."
            parts = folder.split("_")
            if len(parts) < 3:
                logger.warning("Folder name %s does not follow expected format. Skipping.", folder)
                continue
            model_variant = parts[1].lower()
            pretrained_flag = True if parts[2].lower() == "pretrained" else False
            logger.info("Extracted model config for folder %s: variant=%s, pretrained=%s",
                        folder, model_variant, pretrained_flag)

            # Check for the existence of the model file.
            model_file_path = os.path.join(folder_path, "models", "whisper_best.pt")
            if os.path.exists(model_file_path):
                logger.info("Evaluating model from folder: %s", folder)
                # Create the model instance using your provided code.
                model_config = {
                    "pretrained": pretrained_flag,
                    "whisper_variant": model_variant,
                    "num_classes": fixed_config.get("num_classes", 6),
                    "input_channels": fixed_config.get("input_channels", 2)
                }
                model = get_model(model_config)
                state_dict = torch.load(model_file_path, map_location=device)

                # Handle mismatch in number of classes.
                # If the loaded classifier's weight has fewer rows than expected, extend it.
                classifier_weight = state_dict.get("classifier.weight", None)
                if classifier_weight is not None:
                    old_num_classes = classifier_weight.shape[0]
                    new_num_classes = fixed_config.get("num_classes", 6)
                    if old_num_classes != new_num_classes:
                        logger.info("Classifier mismatch: model has %d classes, but expected %d. Adjusting...", 
                                    old_num_classes, new_num_classes)
                        in_features = classifier_weight.shape[1]
                        new_weight = torch.zeros(new_num_classes, in_features)
                        new_bias = torch.zeros(new_num_classes)
                        # Copy over the available weights.
                        new_weight[:old_num_classes] = classifier_weight
                        if "classifier.bias" in state_dict:
                            old_bias = state_dict["classifier.bias"]
                            new_bias[:old_num_classes] = old_bias
                        state_dict["classifier.weight"] = new_weight
                        state_dict["classifier.bias"] = new_bias

                model.load_state_dict(state_dict)
                model.to(device)

                # Evaluate on the source test set.
                source_eval = evaluate_with_confusion_matrix(
                    model, 
                    test_loader, 
                    device, 
                    preprocess_fn=preprocess_whisper_input,
                    save_dir=os.path.join(eval_output_dir, folder, "models_source")
                )
                logger.info("Source test metrics for %s: %s", folder, source_eval)

                # Evaluate LB (65–80 age group) target data.
                lb_train_set, lb_val_set, lb_test_set = preprocess_data_finetune(lb_target_data, lb_target_labels, logger=logger)
                logger.info("65–80 target data: Train set size: %d, Val set size: %d, Test set size: %d",
                            len(lb_train_set), len(lb_val_set), len(lb_test_set))
                lb_test_loader = DataLoader(lb_test_set, batch_size=fixed_config.get("batch_size", 16), shuffle=False)
                lb_eval = evaluate_with_confusion_matrix(
                    model, 
                    lb_test_loader, 
                    device, 
                    preprocess_fn=preprocess_whisper_input,
                    save_dir=os.path.join(eval_output_dir, folder, "models_lb")
                )
                logger.info("65–80 target test metrics for %s: %s", folder, lb_eval)

                # Evaluate LB actual test data.
                lb_test_set_actual, _, _ = preprocess_data_finetune(lb_test_data, lb_test_labels, train_split=1, val_split=0, logger=None)
                lb_test_actual_loader = DataLoader(lb_test_set_actual, batch_size=fixed_config.get("batch_size", 16), shuffle=False)
                lb_actual_eval = evaluate_with_confusion_matrix(
                    model, 
                    lb_test_actual_loader, 
                    device, 
                    preprocess_fn=preprocess_whisper_input,
                    save_dir=os.path.join(eval_output_dir, folder, "models_lb_actual")
                )
                logger.info("65–80 actual test metrics for %s: %s", folder, lb_actual_eval)

                # Evaluate FN (80+ age group) target data.
                fn_train_set, fn_val_set, fn_test_set = preprocess_data_finetune(fn_target_data, fn_target_labels, logger=logger)
                logger.info("80+ target data: Train set size: %d, Val set size: %d, Test set size: %d",
                            len(fn_train_set), len(fn_val_set), len(fn_test_set))
                fn_test_loader = DataLoader(fn_test_set, batch_size=fixed_config.get("batch_size", 16), shuffle=False)
                fn_eval = evaluate_with_confusion_matrix(
                    model, 
                    fn_test_loader, 
                    device, 
                    preprocess_fn=preprocess_whisper_input,
                    save_dir=os.path.join(eval_output_dir, folder, "models_fn")
                )
                logger.info("80+ target test metrics for %s: %s", folder, fn_eval)

                # Evaluate FN actual test data.
                fn_test_set_actual, _, _ = preprocess_data_finetune(fn_test_data, fn_test_labels, train_split=1, val_split=0, logger=None)
                fn_test_actual_loader = DataLoader(fn_test_set_actual, batch_size=fixed_config.get("batch_size", 16), shuffle=False)
                fn_actual_eval = evaluate_with_confusion_matrix(
                    model, 
                    fn_test_actual_loader, 
                    device, 
                    preprocess_fn=preprocess_whisper_input,
                    save_dir=os.path.join(eval_output_dir, folder, "models_fn_actual")
                )
                logger.info("80+ actual test metrics for %s: %s", folder, fn_actual_eval)

                # Save all evaluation metrics for this model.
                all_eval_metrics[folder] = {
                    "source": source_eval,
                    "lb_target": lb_eval,
                    "lb_actual": lb_actual_eval,
                    "fn_target": fn_eval,
                    "fn_actual": fn_actual_eval
                }
            else:
                logger.warning("Skipping folder %s: model file not found.", folder)
        else:
            logger.debug("Skipping non-directory item: %s", folder)

    # Save all evaluation metrics to a pickle file.
    results_file = os.path.join(eval_output_dir, "cross_dataset_evaluation_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(all_eval_metrics, f)
    logger.info("Experiment completed. Evaluation results saved to %s", results_file)

if __name__ == "__main__":
    main()
