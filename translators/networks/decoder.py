from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from translators.utils.torch_utils import init_xavier_uniform, init_lstm_weights


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        enc_lstm_layers: int,
        enc_lstm_bidirec: bool,
        enc_lstm_hidden_dim: int,
        bos_tok_id: int,
        use_stepwise_ctxt: bool = False
    ):
        """
        Initialize the decoder in the sequence-to-sequence translation network.

        Args:
            vocab_size (int): Vocabulary size for the target language.
            embedding_dim (int): Dimension of the embeddings.
            enc_lstm_layers (int): Number of layers in the stacked LSTM for the encoder.
            enc_lstm_hidden_dim (int): Hidden dimensions of the LSTM.
            bos_tok_id (int): Token id for the beginning of sentence token <bos>.
            use_stepwise_ctxt (bool): Whether to provide the encoder hidden state as context for the decoder at each step.
        """
        super().__init__()

        # embeddings
        tok_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.tok_embeddings = init_xavier_uniform(tok_embeddings)

        if use_stepwise_ctxt:
            lstm_in_dim = (enc_lstm_layers * (2 if enc_lstm_bidirec else 1) * enc_lstm_hidden_dim) + embedding_dim
        else:
            lstm_in_dim = embedding_dim
            pass

        # lstm
        lstm_hidden_dim = enc_lstm_hidden_dim * (2 if enc_lstm_bidirec else 1)
        lstm = nn.LSTM(
            input_size=lstm_in_dim,
            num_layers=enc_lstm_layers,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm = init_lstm_weights(lstm)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_size)

        self._lstm_in_ctxt_dim = lstm_in_dim - embedding_dim
        self._bos_tok_id = bos_tok_id
        self._use_stepwise_ctxt = use_stepwise_ctxt
        pass

    @staticmethod
    def _reshape_encoder_state(lstm_state: torch.Tensor) -> torch.Tensor:
        """
        Reshape the encoder LSTM state to make it compatible with the decoeder dimensions.

        Args:
            lstm_state (torch.Tensor): Reshaped state of the encoder.

        Returns:
            torch.Tensor
        """
        reshaped_state = lstm_state.view(
            lstm_state.shape[0] // 2, 2, lstm_state.size(1), lstm_state.size(2)
        )

        reshaped_state = reshaped_state.transpose(1, 2).contiguous()

        reshaped_state = reshaped_state.view(
            lstm_state.size(0) // 2, lstm_state.size(1), lstm_state.size(2) * 2
        )

        return reshaped_state

    def forward(
        self,
        enc_lstm_hidden: torch.Tensor,
        enc_lstm_ctxt: torch.Tensor,
        device: torch.device,
        use_teacher_forcing: bool = False,
        minibatch_target: torch.Tensor = None,
        max_sequence_length: int = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the decoder.

        Args:
            enc_lstm_hidden (torch.Tensor): Final hidden state of the encoder LSTM.
            enc_lstm_ctxt (torch.Tensor): Final context of the encoder LSTM.
            device (torch.device): Device on which to conduct the forward pass.
            use_teacher_forcing (bool): Whether to use teacher forcing for training the decoder.
            minibatch_target (torch.Tensor): Minibatch in the target language, required if `use_teacher_forcing=True`
            max_sequence_length (torch.Tensor): Max sequence length during prediction.

        Returns:
            Tuple[torch.Tensor, Dict]
        """
        if (use_teacher_forcing and minibatch_target is None) and max_sequence_length is None:
            raise ValueError(f"`minibatch_target` required when `use_teacher_forcing=True` or `max_sequence_length` "
                             f"expected.")
            pass

        # batch size
        batch_size = enc_lstm_hidden.shape[1]

        # lstm hidden & ctxt
        lstm_hidden = self._reshape_encoder_state(enc_lstm_hidden).to(device)
        lstm_ctxt = self._reshape_encoder_state(enc_lstm_ctxt).to(device)

        # prediction ctxt
        enc_final_hidden_t = enc_lstm_hidden.transpose(0, 1)
        lstm_in_ctxt = enc_final_hidden_t.reshape(
            batch_size, 1, enc_final_hidden_t.shape[1] * enc_final_hidden_t.shape[2]
        ).contiguous().to(device)

        # start <bos>
        prev_token = torch.empty(batch_size, 1, dtype=torch.long).fill_(self._bos_tok_id).detach().to(device)
        pred_steps = minibatch_target.shape[1]

        # decoder output
        out = list()

        for t in range(pred_steps):
            if self._use_stepwise_ctxt:
                tok_emb = self.tok_embeddings(prev_token)
                lstm_in_ctxt_t = lstm_in_ctxt.broadcast_to(batch_size, tok_emb.shape[1], self._lstm_in_ctxt_dim)
                lstm_in = torch.cat((tok_emb, lstm_in_ctxt_t), dim=2)
            else:
                lstm_in = self.tok_embeddings(prev_token)
                pass

            lstm_out, (lstm_hidden, lstm_ctxt) = self.lstm(
                lstm_in, (lstm_hidden, lstm_ctxt)
            )

            lstm_out_tanh = torch.tanh(lstm_out)
            logits = self.fc(lstm_out_tanh)
            out.append(logits)

            if use_teacher_forcing:
                prev_token = minibatch_target[:, t].unsqueeze(1).detach().to(device)
            else:
                _, prev_token = logits.topk(1)
                prev_token = prev_token.squeeze(-1).detach().to(device)
                pass

        out = torch.cat(out, dim=1)

        return out, {
            "lstm_in": lstm_in.detach(),
            "lstm_hidden": lstm_hidden.detach(),
            "lstm_ctxt": lstm_ctxt.detach(),
            "lstm_out": lstm_out.detach(),
            "lstm_out_tanh": lstm_out_tanh.detach(),
            "logits": out.detach(),
        }

    def infer(
        self,
        enc_lstm_hidden: torch.Tensor,
        enc_lstm_ctxt: torch.Tensor,
        max_sequence_length: int,
        device: torch.device
    ) -> List:
        """
        Generate a translated output.

        Args:
            enc_lstm_hidden (torch.Tensor): Final hidden state of the encoder LSTM.
            enc_lstm_ctxt (torch.Tensor): Final context of the encoder LSTM.
            max_sequence_length (torch.Tensor): Max sequence length during prediction.
            device (torch.device): Device on which to run inference.

        Returns:
            Tuple[torch.Tensor, Dict]
        """
        # set to "1" for inference.
        batch_size = enc_lstm_hidden.shape[1]
        g = list()

        if batch_size != 1:
            raise ValueError("`infer` only operates on `batch_size` of 1.")
            pass

        # lstm hidden & ctxt
        lstm_hidden = self._reshape_encoder_state(enc_lstm_hidden).to(device)
        lstm_ctxt = self._reshape_encoder_state(enc_lstm_ctxt).to(device)

        # prediction ctxt
        enc_final_hidden_t = enc_lstm_hidden.transpose(0, 1)
        lstm_in_ctxt = enc_final_hidden_t.reshape(
            batch_size, 1, enc_final_hidden_t.shape[1] * enc_final_hidden_t.shape[2]
        ).contiguous()

        # start <bos>
        prev_token = torch.empty(batch_size, 1, dtype=torch.long).fill_(self._bos_tok_id).detach().to(device)
        pred_steps = max_sequence_length

        for t in range(pred_steps):
            if self._use_stepwise_ctxt:
                tok_emb = self.tok_embeddings(prev_token)
                lstm_in_ctxt_t = lstm_in_ctxt.broadcast_to(batch_size, tok_emb.shape[1], self._lstm_in_ctxt_dim)
                lstm_in = torch.cat((tok_emb, lstm_in_ctxt_t), dim=2)
            else:
                lstm_in = self.tok_embeddings(prev_token)
                pass

            lstm_out, (lstm_hidden, lstm_ctxt) = self.lstm(
                lstm_in, (lstm_hidden, lstm_ctxt)
            )

            lstm_out_tanh = torch.tanh(lstm_out)
            logits = self.fc(lstm_out_tanh)
            log_probs = F.softmax(logits, dim = -1)

            # inference
            next_token = torch.multinomial(log_probs.squeeze(1), num_samples = 1)
            g.extend(next_token.tolist()[0])
            prev_token = next_token.detach().to(device)
            pass

        return g
