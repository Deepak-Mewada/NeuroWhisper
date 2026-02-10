"""
eval.py

This module provides evaluation functions for your models.
It computes metrics, plots ROC curves, and includes a preprocessing function
to prepare data for the Whisper model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torchaudio.transforms as T

def preprocess_whisper_input(flattened_data, sampling_rate=16000, device="cpu"):
    """
    Preprocess data for Whisper by normalizing and converting to a Mel spectrogram.
    """
    normalized_data = (flattened_data - flattened_data.mean(dim=1, keepdim=True)) / flattened_data.std(dim=1, keepdim=True)
    mel_transform = T.MelSpectrogram(sample_rate=sampling_rate, n_fft=400, hop_length=160, n_mels=80).to(device)
    mel_spectrogram = mel_transform(normalized_data)
    expected_length = 3000
    if mel_spectrogram.shape[-1] < expected_length:
        mel_spectrogram = F.pad(mel_spectrogram, (0, expected_length - mel_spectrogram.shape[-1]))
    elif mel_spectrogram.shape[-1] > expected_length:
        mel_spectrogram = mel_spectrogram[..., :expected_length]
    return mel_spectrogram


import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score
)

# ======================
# Metrics & Plotting Utils
# ======================

def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, preds, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(labels, preds),
    }

def plot_roc_curve(all_labels, all_probs, num_classes, save_path):
    unique_classes = np.unique(all_labels)
    if len(unique_classes) < 2:
        print("Skipping ROC curve: Only one class present in y_true.")
        return
    plt.figure(figsize=(8, 5))
    for i in range(num_classes):
        if i not in unique_classes:
            print(f"Skipping class {i}: Not present in y_true.")
            continue
        try:
            fpr, tpr, _ = roc_curve((np.array(all_labels) == i).astype(int), all_probs[:, i])
            auc_score = roc_auc_score((np.array(all_labels) == i).astype(int), all_probs[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
        except ValueError as e:
            print(f"Skipping class {i} due to error: {e}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(all_labels, all_probs, num_classes, save_path):
    unique_classes = np.unique(all_labels)
    if len(unique_classes) < 2:
        print("Skipping Precision-Recall curve: Only one class present in y_true.")
        return
    plt.figure(figsize=(8, 5))
    for i in range(num_classes):
        if i not in unique_classes:
            print(f"Skipping class {i}: Not present in y_true.")
            continue
        try:
            precision, recall, _ = precision_recall_curve((np.array(all_labels) == i).astype(int), all_probs[:, i])
            ap_score = average_precision_score((np.array(all_labels) == i).astype(int), all_probs[:, i])
            plt.plot(recall, precision, label=f'Class {i} (AP = {ap_score:.2f})')
        except ValueError as e:
            print(f"Skipping class {i} due to error: {e}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_loss_curves(epochs, train_losses, val_losses, train_accuracies, val_accuracies, save_dir, model_name):
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss_curve.png"))
    plt.close()
    
    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_accuracy_curve.png"))
    plt.close()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_tsne(embeddings, labels, save_path="tsne_plot.png"):
    """
    Plot a t-SNE visualization of feature embeddings and save the plot.
    
    Args:
        embeddings (np.array): 2D array with shape (n_samples, n_features).
        labels (np.array or list): Corresponding labels for each sample.
        save_path (str): File path where the t-SNE plot will be saved.
    """
    # Ensure embeddings is 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    n_samples = embeddings.shape[0]
    # Adjust perplexity: it must be less than n_samples. Default TSNE uses perplexity=30.
    perplexity = 30 if n_samples > 30 else max(1, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1],
                    hue=labels, palette="viridis", legend="full", alpha=0.7)
    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pca(embeddings, labels, save_path="pca_plot.png"):
    """
    Plot a PCA visualization of feature embeddings and save the plot.
    
    Args:
        embeddings (np.array): 2D array with shape (n_samples, n_features).
        labels (np.array or list): Corresponding labels for each sample.
        save_path (str): File path where the PCA plot will be saved.
    """
    # Ensure embeddings is 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    n_samples, n_features = embeddings.shape
    if n_samples < 2 or n_features < 2:
        print("Not enough samples or features for PCA; skipping PCA plot.")
        return

    # Perform PCA to reduce dimensions to 2
    try:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(embeddings)
    except Exception as e:
        print(f"PCA computation failed: {e}")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1],
                    hue=labels, palette="viridis", legend="full", alpha=0.7)
    plt.title("PCA Visualization of Feature Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to {save_path}")



import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, input_tensor, layer_idx=0, head_idx=0, save_path="attention_plot.png", max_tokens=None):
    """
    Visualize the attention weights for a specified layer and head.
    
    Args:
        model: Transformer-based model (e.g., Whisper) configured to output attentions.
        input_tensor: Input tensor with shape (batch, sequence_length, features).
        layer_idx (int): Index of the encoder layer to inspect.
        head_idx (int): Index of the attention head within that layer.
        save_path (str): File path to save the attention plot.
        max_tokens (int, optional): If provided, only the first `max_tokens` tokens are visualized.
    """
    # Ensure model outputs attentions
    try:
        outputs = model(input_tensor, output_attentions=True)
    except Exception as e:
        print("Error calling model with output_attentions=True:", e)
        return
    
    if not hasattr(outputs, "attentions"):
        print("Model did not return attentions; please check your model configuration.")
        return
    
    attentions = outputs.attentions  # Expected to be a tuple of attention matrices
    if layer_idx >= len(attentions):
        print(f"Requested layer_idx {layer_idx} exceeds the number of layers ({len(attentions)}).")
        return
    
    attn_matrix = attentions[layer_idx]  # Shape: (batch, num_heads, seq_len, seq_len)
    if attn_matrix.ndim < 4:
        print("Attention matrix does not have expected 4D shape.")
        return
    
    # Check if the requested head index is valid.
    if head_idx >= attn_matrix.shape[1]:
        print(f"Requested head_idx {head_idx} exceeds available heads ({attn_matrix.shape[1]}).")
        return
    
    # Use the first sample in the batch.
    attn_matrix = attn_matrix[0, head_idx].detach().cpu().numpy()
    
    # Print shape to debug if needed.
    print("Attention matrix shape:", attn_matrix.shape)
    
    # Optionally subsample the matrix if max_tokens is provided.
    if max_tokens is not None and attn_matrix.shape[0] > max_tokens:
        attn_matrix = attn_matrix[:max_tokens, :max_tokens]
        print("Subsampled attention matrix shape:", attn_matrix.shape)
    
    plt.figure(figsize=(10, 8))
    # Disable annotation to speed up rendering.
    sns.heatmap(attn_matrix, cmap='viridis', annot=False)
    plt.title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Attention plot saved to {save_path}")

    
def visualize_attention_from_loader(model, loader, device, preprocess_fn=None, layer_idx=0, head_idx=0, save_path="attention_plot.png", max_tokens=None):
    """
    Extracts one batch from the loader, takes the first sample, and visualizes its attention.
    
    Args:
        model: The transformer model configured to output attentions.
        loader: DataLoader returning (inputs, labels).
        device: The device (e.g., 'cpu' or 'cuda').
        preprocess_fn: Optional preprocessing function to transform inputs.
        layer_idx: Index of the layer from which to extract attention.
        head_idx: Index of the head within that layer.
        save_path: File path to save the attention plot.
        max_tokens (int, optional): Maximum number of tokens to visualize.
    """
    # Get one batch from the loader
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    if preprocess_fn:
        inputs = preprocess_fn(inputs, device=device)
    # Select the first sample in the batch.
    sample = inputs[0:1]
    
    # Now call the visualize_attention function on this sample.
    visualize_attention(model, sample, layer_idx=layer_idx, head_idx=head_idx, save_path=save_path, max_tokens=max_tokens)

# ======================
# Evaluate Function
# ======================

def evaluate(model, loader, device, preprocess_fn=None, num_classes=6, save_dir = 'outputs'):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if preprocess_fn:
                inputs = preprocess_fn(inputs, device=device)
            outputs,pooled = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs)
            print(f"outputs : {outputs}")
            # pooled = outputs.mean(dim=1)
            all_embeddings.append(pooled.cpu().numpy())
    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)
    metrics["accuracy"] = correct / total_samples
    # Save ROC and PR curves
    roc_path = os.path.join(save_dir, "roc_curve.png")
    pr_path = os.path.join(save_dir, "pr_curve.png")
    tsne_path = os.path.join(save_dir, "tSNE_plot.png")
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plot_roc_curve(np.array(all_labels), all_probs, num_classes, roc_path)
    plot_precision_recall_curve(np.array(all_labels), all_probs, num_classes, pr_path)
    embeddings = np.concatenate(all_embeddings, axis=0)
    # print(f"embeddings : {embeddings}")
    plot_tsne(embeddings, all_labels, save_path=tsne_path)
    pca_path = os.path.join(save_dir, "pca_plot.png")
    plot_pca(embeddings, all_labels, save_path=pca_path)
    att_path = os.path.join(save_dir, "attention_visualization.png")
    visualize_attention_from_loader(model, loader, device, preprocess_fn, layer_idx=0, head_idx=0, save_path=att_path)
    
    # Also return detailed evaluation outputs for further analysis
    return metrics, np.array(all_labels), np.array(all_preds), all_probs

	