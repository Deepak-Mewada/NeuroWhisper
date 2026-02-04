import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig
from types import SimpleNamespace  # For returning multiple outputs

class WhisperForEEGClassification_pretrined(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(WhisperForEEGClassification_pretrined, self).__init__()
        # Load the pretrained model with the manual attention implementation.
        self.whisper = WhisperModel.from_pretrained(pretrained_model_name, attn_implementation="eager")
        # Ensure the model is configured to output attentions.
        self.whisper.config.output_attentions = True
        self.classifier = nn.Linear(self.whisper.config.d_model, num_classes)
        
    def forward(self, input_features, output_attentions=False):
        outputs = self.whisper.encoder(input_features, output_attentions=output_attentions)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        if output_attentions:
            return SimpleNamespace(logits=logits, attentions=outputs.attentions, embeddings=pooled_output)
        return logits, pooled_output

class WhisperForEEGClassification_without_pretrained(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(WhisperForEEGClassification_without_pretrained, self).__init__()
        # Load the configuration for the specified Whisper variant.
        config = WhisperConfig.from_pretrained(pretrained_model_name)
        # Ensure attentions are output by default.
        config.output_attentions = True
        # Specify the manual attention implementation to remove the warning.
        config.attn_implementation = "eager"
        # Initialize the Whisper model from scratch with random weights.
        self.whisper = WhisperModel(config)
        self.classifier = nn.Linear(config.d_model, num_classes)
        
    def forward(self, input_features, output_attentions=False):
        outputs = self.whisper.encoder(input_features, output_attentions=output_attentions)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        if output_attentions:
            return SimpleNamespace(logits=logits, attentions=outputs.attentions, embeddings=pooled_output)
        return logits, pooled_output

def get_model(config, logger=None):
    """
    Constructs and returns a Whisper model for EEG classification based on the configuration.
    
    Expected keys in config:
      - num_classes: Number of output classes (default: 6).
      - pretrained: Boolean flag to load pretrained weights (default: False).
      - whisper_variant: Desired Whisper model variant ("tiny", "base", "small", "medium", "large"; default: "base").
      - input_channels: Number of channels in the input data (default: 2).
    """
    num_classes = config.get("num_classes", 6)
    variant = config.get("whisper_variant", "base")
    pretrained = config.get("pretrained", False)
    input_channels = config.get("input_channels", 2)
    
    if logger:
        logger.info(
            f"Loading Whisper model variant: {variant} with "
            f"{'pretrained weights' if pretrained else 'random initialization'}, "
            f"input_channels: {input_channels}"
        )
    
    if pretrained:
        return WhisperForEEGClassification_pretrined(pretrained_model_name=f"openai/whisper-{variant}", num_classes=num_classes)
    else:
        return WhisperForEEGClassification_without_pretrained(pretrained_model_name=f"openai/whisper-{variant}", num_classes=num_classes)
