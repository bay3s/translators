import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def init_module(
    module: nn.Module,
    weight_init: callable,
    bias_init: callable = None,
    gain: float = 1.0,
) -> nn.Module:
    """
    Initialize a module given the weight initialization scheme, bias initialization, and gain.

    Args:
        module (nn.Module):
        weight_init (callable):
        bias_init (callable):
        gain (float):

    Returns:
        nn.Module
    """
    weight_init(module.weight.data, gain=gain)

    if bias_init is not None:
        bias_init(module.bias.data)
        pass

    weight_norm(module)

    return module


def init_xavier_uniform(m: nn.Module) -> nn.Module:
    """
    Applies Xavier initialization to the weights of the neural net.

    Args:
        m (nn.Module): Module with neural net weights with Xavier initialization.

    Returns:
        nn.Module
    """
    bias_init = (lambda x: nn.init.constant_(x, 0)) if hasattr(m, "bias") else None

    return init_module(m, nn.init.xavier_uniform_, bias_init, np.sqrt(2))


def init_lstm_weights(lstm: torch.nn.LSTM) -> torch.nn.LSTM:
    """
    Initialize weights for the given LSTM.

    Args:
        lstm (torch.nn.LSTM) LSTM network for which to initialize weights.

    Returns:
        torch.nn.LSTM
    """
    for name, param in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param)

    return lstm


def init_embedding_weights(embeddings: torch.nn.Embedding) -> torch.nn.Embedding:
    """
    Initialize embedding weights.

    Args:
        embeddings (nn.Embeddings): Initialize embedding weights.

    Returns:
        nn.Embedding
    """
    return init_xavier_uniform(embeddings)
