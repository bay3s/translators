from typing import Tuple, Dict
import torch
import torch.nn as nn

from translators.utils.torch_utils import init_embedding_weights, init_lstm_weights


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_num_layers: int,
        lstm_hidden_dim: int,
        lstm_bidirec: bool,
        dropout_p: float,
    ):
        """
        Initialize the encoder.

        Args:
            vocab_size (int): Size of the source language vocabulary.
            embedding_dim (int): Dimension of the token embeddings.
            lstm_num_layers (int): Number of LSTM layers to stack.
            lstm_hidden_dim (int): Dimensions of the LSTM hidden state.
            lstm_bidirec (bool): Whether to use the bidirectional LSTM.
            dropout_p (float): Probabiltiy of dropout.
        """
        super().__init__()

        tok_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.tok_embeddings = init_embedding_weights(tok_embeddings)
        self.emb_dropout = nn.Dropout(dropout_p)

        lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirec,
            batch_first=True,
        )

        self.lstm = init_lstm_weights(lstm)
        pass

    def forward(
        self, source_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Give a minibatch of data from the source language, returns encoded representations, as well as the outputs
        at each layer in the neural network.

        Args:
            source_batch (torch.Tensor): Minibatch of source language data to encode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        # embeddings
        emb_out = self.tok_embeddings(source_batch)
        emb_out_tanh = torch.tanh(emb_out)
        emb_tanh_dropout = self.emb_dropout(emb_out_tanh)

        # dims
        batch_size = emb_tanh_dropout.shape[0]
        timesteps = emb_tanh_dropout.shape[1]
        embedding_dim = emb_tanh_dropout.shape[2]

        # forward
        lstm_in = emb_tanh_dropout.view(batch_size, timesteps, embedding_dim)
        lstm_out, (lstm_hidden, lstm_ctxt) = self.lstm(lstm_in)

        return (
            lstm_hidden,
            lstm_ctxt,
            {
                "emb_out": emb_out.detach(),
                "emb_out_tanh": emb_out_tanh.detach(),
                "emb_tanh_dropout": emb_tanh_dropout.detach(),
                "lstm_in": lstm_in.detach(),
                "lstm_hidden": lstm_hidden.detach(),
                "lstm_ctxt": lstm_ctxt.detach(),
                "lstm_out": lstm_out.detach(),
            },
        )
