from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.vocab import Vocab

import wandb

from translators.utils.constants import Constants
from translators.datasets.translation_dataset import TranslationDataset

from translators.networks.encoder import Encoder
from translators.networks.decoder import Decoder


class Trainer:
    def __init__(
        self,
        source_language: str,
        target_language: str,
        batch_size: int,
        enc_learning_rate: float,
        dec_learning_rate: float,
        num_epochs: int,
        device: torch.device
    ):
        """
        Provides a wrapper for parameterizing and logging training runs.

        Args:
            source_language (str): Source language for the translation model.
            target_language (str): Target language for the translation model.
            batch_size (int): Training batch size for the model.
            enc_learning_rate (float): Learning rate for the encoder.
            dec_learning_rate (float): Learning rate for the decoder.
            num_epochs (int): Number of epochs to train this model for.
            device (torch.device): Device on which to train the model.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        # dataset
        self.lang_dataset = TranslationDataset(source_language, target_language)

        # loaders
        self.train_loader, (
            self.source_vocab,
            self.target_vocab,
        ) = self.lang_dataset.load("train", batch_size, device)

        self.val_loader, _ = self.lang_dataset.load("val", batch_size, device)

        # @todo accept all these as parameters.
        self.enc, self.dec = self.init_networks(
            self.source_vocab,
            self.target_vocab,
            enc_embedding_dim=256,
            dec_embedding_dim=256,
            lstm_num_layers=4,
            lstm_hidden_dim=256,
            use_bidirec_lstm=True,
            dropout_p=0.025,
            device=device,
        )

        # opt
        self.enc_opt = torch.optim.AdamW(self.enc.parameters(), lr=enc_learning_rate)
        self.dec_opt = torch.optim.AdamW(self.dec.parameters(), lr=dec_learning_rate)
        self.loss_function = nn.NLLLoss()
        pass

    def train(self) -> None:
        """
        Run the training loop.

        Returns:
            None
        """
        wandb.login()
        logs_directory = "./logs"
        wandb.init(project = f"translators", dir = Path(logs_directory).mkdir(parents = True, exist_ok = True))

        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch + 1}/{self.num_epochs}]")

            torch.enable_grad()
            self.enc.train()
            self.dec.train()

            train_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # sample inference
                with torch.no_grad():
                    b_source_i = inputs[0]
                    lstm_hidden, lstm_ctxt, _ = self.enc(b_source_i.unsqueeze(0))
                    g = self.dec.infer(lstm_hidden, lstm_ctxt, 20, self.device)
                    print(" ".join(np.array(self.target_vocab.get_itos())[g]))
                    pass

                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()

                lstm_hidden, lstm_ctxt, _ = self.enc(inputs)
                logits, _ = self.dec(
                    lstm_hidden,
                    lstm_ctxt,
                    self.device,
                    use_teacher_forcing=True,
                    minibatch_target=targets,
                )

                # loss
                logprobs = F.log_softmax(logits, dim = -1).to(self.device)
                loss = self.loss_function(logprobs.view(-1, logprobs.size(-1)), targets.view(-1))
                loss.backward()

                # opt
                self.enc_opt.step()
                self.dec_opt.step()
                train_loss += loss.item()
                pass


            train_loss /= len(self.train_loader)

            wandb_logs = dict()
            wandb_logs["train_loss"] = train_loss
            wandb.log(wandb_logs)
            pass

    @staticmethod
    def init_networks(
        source_vocab: Vocab,
        target_vocab: Vocab,
        enc_embedding_dim: int,
        dec_embedding_dim: int,
        lstm_num_layers: int,
        lstm_hidden_dim: int,
        use_bidirec_lstm: bool,
        dropout_p: float,
        device: torch.device,
    ) -> Tuple[Encoder, Decoder]:
        """
        Set up the encoder-decoder networks.

        Args:
            source_vocab (Vocab): Source vocabulary.
            target_vocab (Vocab): Target vocabulary.
            enc_embedding_dim (int): Embedding dimensions for the encoder.
            dec_embedding_dim (int): Embedding dimensions for the decoder.
            lstm_num_layers (int): Number of stacked layers in the LSTM.
            lstm_hidden_dim (int): Number of hidden dimensions in the LSTM.
            use_bidirec_lstm (bool): Whether to use a bidirectional LSTM.
            dropout_p (float): Probability of dropouts.
            device (torch.device): Device to run training on.

        Returns:
            Tuple[Encoder, Decoder]
        """
        enc = Encoder(
            source_vocab_size=len(source_vocab.get_itos()),
            embedding_dim=enc_embedding_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_bidirec=use_bidirec_lstm,
            dropout_p=dropout_p,
        ).to(device)

        dec = Decoder(
            target_vocab_size=len(target_vocab.get_itos()),
            embedding_dim=dec_embedding_dim,
            enc_lstm_hidden_dim=lstm_hidden_dim,
            enc_lstm_bidirec=use_bidirec_lstm,
            enc_lstm_layers=lstm_num_layers,
            bos_tok_id=source_vocab.get_stoi()[Constants.SPECIAL_TOKEN_BOS],
            use_stepwise_ctxt=True,
        ).to(device)

        return enc, dec

    def save_checkpoint(
        self, checkpoint_path: str, train_loss: float, val_loss: float
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            checkpoint_path (str): Path where to save the checkpoint.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.

        Returns:
            None
        """
        torch.save(
            {
                "epoch": self.num_epochs,
                "enc_state_dict": self.enc.state_dict(),
                "enc_optimizer_state_dict": self.enc_opt.state_dict(),
                "dec_state_dict": self.dec.state_dict(),
                "dec_optimizer_state_dict": self.dec_opt.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
