"""
main.py

This is the main runner script. It loads configuration from a YAML file,
sets up logging and a unique output directory, and executes the full pipeline:
data preprocessing, model building, training, evaluation, and fine-tuning.
"""

import os
import sys
import yaml
import logging
import pickle
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import random
from preprocessing.preprocess import get_phase_1_data, get_shape, preprocess_data, preprocess_data_finetune
from model.model_builder import get_model
from trainandeval.train import train_and_evaluate
from trainandeval.eval import evaluate, preprocess_whisper_input

def setup_logging(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = os.path.join(output_dir, "run.log")
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def main():
    # Load configuration from YAML file (default: config.yaml or pass as an argument)
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Create a unique output folder based on run_name and timestamp
    run_name = config.get("run_name", "default_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)


    logger = setup_logging(output_dir)
    logger.info("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Starting experiment with configuration: %s", config)
    logger.info("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    # Load phase 1 data from all sources using preprocessing module.
    source_data, source_labels, lb_target_data, lb_target_labels, lb_test_data, lb_test_labels, fn_target_data, fn_target_labels, fn_test_data, fn_test_labels = get_phase_1_data(config, logger)
    
    
    # Optionally load test labels from files if provided in config.
    if "LB_TEST_LABELS_FILE" in config:
        import numpy as np
        lb_test_labels = np.load(config["LB_TEST_LABELS_FILE"])
    if "FN_TEST_LABELS_FILE" in config:
        import numpy as np
        fn_test_labels = np.load(config["FN_TEST_LABELS_FILE"])
        

#######################################################################################################

    # import numpy as np

    # # Function to generate random EEG data (2 channels, 3000 samples)
    # def generate_eeg_data(num_samples):
    #     return [np.random.randn(2, 3000).astype(np.float32) for _ in range(num_samples)]

    # # Function to generate random labels (6-class classification)
    # def generate_labels(num_samples):
    #     return np.random.randint(0, 6, size=(num_samples,))

    # # Generate small datasets
    # source_data = generate_eeg_data(10)
    # source_labels = generate_labels(10)

    # lb_target_data = generate_eeg_data(10)
    # lb_target_labels = generate_labels(10)

    # lb_test_data = generate_eeg_data(10)
    # lb_test_labels = generate_labels(10)

    # fn_target_data = generate_eeg_data(10)
    # fn_target_labels = generate_labels(10)

    # fn_test_data = generate_eeg_data(10)
    # fn_test_labels = generate_labels(10)

    # # Print shapes to verify
    # def get_shape(data):
    #     if isinstance(data, np.ndarray):
    #         return data.shape
    #     elif isinstance(data, list):
    #         return len(data), "(List - First Element Shape: {})".format(np.array(data[0]).shape if len(data) > 0 and isinstance(data[0], (list, np.ndarray)) else "N/A")
    #     else:
    #         return "Unknown Type"

    # print("source_data shape:", get_shape(source_data))
    # print("source_labels shape:", get_shape(source_labels))
    # print("lb_target_data shape:", get_shape(lb_target_data))
    # print("lb_target_labels shape:", get_shape(lb_target_labels))
    # print("lb_test_data shape:", get_shape(lb_test_data))
    # print("lb_test_labels shape:", get_shape(lb_test_labels))
    # print("fn_target_data shape:", get_shape(fn_target_data))
    # print("fn_target_labels shape:", get_shape(fn_target_labels))
    # print("fn_test_data shape:", get_shape(fn_test_data))
    # print("fn_test_labels shape:", get_shape(fn_test_labels))

 #######################################################################################################   

    # source_data = source_data[:50]
    # source_labels = source_labels[:50]
    # lb_target_data = lb_target_data[:50]
    # lb_target_labels = lb_target_labels[:50]
    # lb_test_data = lb_test_data[:50]
    # lb_test_labels = lb_test_labels[:50]
    # fn_target_data = fn_target_data[:50]
    # fn_target_labels = fn_target_labels[:50]
    # fn_test_data = fn_test_data[:50]
    # fn_test_labels = fn_test_labels[:50]

    
        # Log shapes of the datasets.
    logger.info("source_data shape: %s", get_shape(source_data))
    logger.info("source_labels shape: %s", get_shape(source_labels))
    logger.info("lb_target_data shape: %s", get_shape(lb_target_data))
    logger.info("lb_target_labels shape: %s", get_shape(lb_target_labels))
    logger.info("lb_test_data shape: %s", get_shape(lb_test_data))
    logger.info("lb_test_labels shape: %s", get_shape(lb_test_labels))
    logger.info("fn_target_data shape: %s", get_shape(fn_target_data))
    logger.info("fn_target_labels shape: %s", get_shape(fn_target_labels))
    logger.info("fn_test_data shape: %s", get_shape(fn_test_data))
    logger.info("fn_test_labels shape: %s", get_shape(fn_test_labels))
    
    
    seed = 42  # You can choose any integer value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    # Preprocess the source data into training, validation, and test sets.
    train_set, val_set, test_set = preprocess_data(source_data, source_labels, logger=logger)
    if logger:
        logger.info(f"Source data: Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")
    else:
        print(f"Source data: Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=config.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.get("batch_size", 16), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.get("batch_size", 16), shuffle=False)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build and train the primary model (e.g., Whisper).
    model_name = config.get("model_type", "whisper")
    model_varient = config.get("whisper_variant", "base")
    model = get_model(config=config, logger=logger).to(device)
    logger.info("Training %s - %s model on source data...", model_name,model_varient)
    
    model, train_losses, val_losses, test_metrics = train_and_evaluate(
        model_name, model, train_loader, val_loader, test_loader, device,
        num_epochs=config.get("num_epochs", 7),
        patience=config.get("patience", 3),
        save_dir=os.path.join(output_dir, "models"),
        preprocess_fn=preprocess_whisper_input,
        logger=logger
    )
    logger.info("Evaluating model on source test data...")
    source_test_metrics = evaluate(model, test_loader, device, preprocess_fn=preprocess_whisper_input, save_dir=os.path.join(output_dir, "models"))
    logger.info("Source test metrics: %s", source_test_metrics)

    # Fine-tune on 65-80 age group target data.
    lb_train_set, lb_val_set, lb_test_set = preprocess_data_finetune(lb_target_data, lb_target_labels, logger=logger)
    if logger:
        logger.info(f"65-80 age group target data: Train set size: {len(lb_train_set)}, Val set size: {len(lb_val_set)}, Test set size: {len(lb_test_set)}")
    else:
        print(f"65-80 age group target data: Train set size: {len(lb_train_set)}, Val set size: {len(lb_val_set)}, Test set size: {len(lb_test_set)}")
    
    lb_train_loader = DataLoader(lb_train_set, batch_size=config.get("batch_size", 16), shuffle=True)
    lb_val_loader = DataLoader(lb_val_set, batch_size=config.get("batch_size", 16), shuffle=False)
    lb_test_loader = DataLoader(lb_test_set, batch_size=config.get("batch_size", 16), shuffle=False)
    logger.info("Fine-tuning on 65-80 age group target data...")
    lb_model = get_model(config=config, logger=logger).to(device)
    lb_model, _, _, lb_test_metrics = train_and_evaluate(
        model_name, lb_model, lb_train_loader, lb_val_loader, lb_test_loader, device,
        num_epochs=config.get("num_epochs", 7),
        patience=config.get("patience", 3),
        save_dir=os.path.join(output_dir, "models_lb"),
        preprocess_fn=preprocess_whisper_input,
        logger=logger
    )
    logger.info("Evaluating fine-tuned model on 65-80 age group test data...")
    lb_test_metrics = evaluate(lb_model, lb_test_loader, device, preprocess_fn=preprocess_whisper_input, save_dir=os.path.join(output_dir, "models_lb"))
    logger.info("65-80 age group test metrics: %s", lb_test_metrics)
    
    
    lb_test_set_actual, _, _ = preprocess_data_finetune(lb_test_data, lb_test_labels,
                                                        train_split=1, val_split=0, logger=None)
    lb_test_set_actual_loader = DataLoader(lb_test_set_actual, batch_size=config.get("batch_size", 16), shuffle=False)
    logger.info("Evaluating fine-tuned model on 65-80 age group(on given actual) test data...")
    lb_test_metrics_actual = evaluate(lb_model, lb_test_set_actual_loader, device, preprocess_fn=preprocess_whisper_input, save_dir=os.path.join(output_dir, "models_lb_actual"))
    logger.info("65-80 age group test metrics (on given actual): %s", lb_test_metrics_actual)
    

    # Fine-tune on 80+ age group target data.
    fn_train_set, fn_val_set, fn_test_set = preprocess_data_finetune(fn_target_data, fn_target_labels, logger=logger)
    if logger:
        logger.info(f"80+ age group target data: Train set size: {len(fn_train_set)}, Val set size: {len(fn_val_set)}, Test set size: {len(fn_test_set)}")
    else:
        print(f"80+ age group target data: Train set size: {len(fn_train_set)}, Val set size: {len(fn_val_set)}, Test set size: {len(fn_test_set)}")
        
    fn_train_loader = DataLoader(fn_train_set, batch_size=config.get("batch_size", 16), shuffle=True)
    fn_val_loader = DataLoader(fn_val_set, batch_size=config.get("batch_size", 16), shuffle=False)
    fn_test_loader = DataLoader(fn_test_set, batch_size=config.get("batch_size", 16), shuffle=False)
    logger.info("Fine-tuning on 80+ age group target data...")
    fn_model = get_model(config=config, logger=logger).to(device)
    fn_model, _, _, fn_test_metrics = train_and_evaluate(
        model_name, fn_model, fn_train_loader, fn_val_loader, fn_test_loader, device,
        num_epochs=config.get("num_epochs", 7),
        patience=config.get("patience", 3),
        save_dir=os.path.join(output_dir, "models_fn"),
        preprocess_fn=preprocess_whisper_input,
        logger=logger
    )
    logger.info("Evaluating fine-tuned model on 80+ age group test data...")
    fn_test_metrics = evaluate(fn_model, fn_test_loader, device, preprocess_fn=preprocess_whisper_input, save_dir=os.path.join(output_dir, "models_fn"))
    logger.info("80+ age group test metrics: %s", fn_test_metrics)
    
    
    fn_test_set_actual, _, _ = preprocess_data_finetune(fn_test_data, fn_test_labels,
                                                        train_split=1, val_split=0, logger=None)
    fn_test_set_actual_loader = DataLoader(fn_test_set_actual, batch_size=config.get("batch_size", 16), shuffle=False)
    logger.info("Evaluating fine-tuned model on 80+ age group(on given actual) test data...")
    fn_test_metrics_actual = evaluate(fn_model, fn_test_set_actual_loader, device, preprocess_fn=preprocess_whisper_input, save_dir=os.path.join(output_dir, "models_fn_actual"))
    logger.info("80+ age group test metrics (on given actual): %s", fn_test_metrics_actual)

    # Save metrics and training histories.
    results = {
        "source_test_metrics": test_metrics,
        "lb_test_metrics": lb_test_metrics,
        "fn_test_metrics": fn_test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    logger.info("Experiment completed. Results saved to %s", output_dir)

if __name__ == "__main__":
    main()
