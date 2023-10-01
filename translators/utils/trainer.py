import random
from pathlib import Path
from typing import Tuple

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
            dropout_p=0.01,
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
        enable_wandb = False
        if enable_wandb:
            wandb.login()
            logs_directory = "./logs"
            wandb.init(project = f"translators", dir = Path(logs_directory).mkdir(parents = True, exist_ok = True))
            pass

        for epoch in range(self.num_epochs):
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

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
                logprobs = F.softmax(logits, dim = -1).to(self.device)
                loss = self.loss_function(logprobs.view(-1, logprobs.size(-1)), targets.view(-1))
                loss.backward()

                # opt
                self.enc_opt.step()
                self.dec_opt.step()
                train_loss += loss.item()
                pass

            train_loss /= len(self.train_loader)
            val_loss = self.estimate_val_loss()

            wandb_logs = dict()
            wandb_logs["train_loss"] = train_loss
            wandb_logs["val_loss"] = val_loss
            wandb.log(wandb_logs)
            pass

    def run_inference(self, b_source_i):
        with torch.no_grad():
            source_words = []

            for tok_idx in b_source_i.clone().to("cpu").squeeze().tolist():
                source_words.append(self.source_vocab.get_itos()[tok_idx])
                pass

            lstm_hidden, lstm_ctxt, _ = self.enc(b_source_i.unsqueeze(0))
            decoded_tok_ids = self.dec.infer(lstm_hidden, lstm_ctxt, 20, self.device)

            decoded_words = []
            for tok_id in decoded_tok_ids:
                decoded_words.append(self.target_vocab.get_itos()[tok_id])
                pass

            log_message = " ".join(decoded_words)
            print("translation output: ", log_message)
            print()
            pass

    def estimate_val_loss(self) -> float:
        """
        Estimate loss using the validation set.

        Returns:
            float
        """
        self.enc.eval()
        self.dec.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # forward
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                lstm_hidden, lstm_ctxt, _ = self.enc(inputs)

                logits, _ = self.dec(
                    lstm_hidden,
                    lstm_ctxt,
                    self.device,
                    use_teacher_forcing = True,
                    minibatch_target = targets,
                )

                # loss
                logprobs = F.softmax(logits, dim = -1).to(self.device)
                loss = self.loss_function(logprobs.view(-1, logprobs.size(-1)), targets.view(-1))
                val_loss += loss.item()
                continue

        val_loss /= len(self.val_loader)

        # inference
        print("with `mode=eval`")
        sampled_idx = [random.randint(0, len(inputs) - 1) for _ in range(int(0.2 * len(inputs)))]
        sampled_inputs = inputs[sampled_idx]
        for i in range(len(sampled_inputs)):
            inputs_i = sampled_inputs[i]
            self.run_inference(inputs_i)
            pass

        self.enc.train()
        self.dec.train()

        print("with `mode=train`")
        for i in range(len(sampled_inputs)):
            inputs_i = sampled_inputs[i]
            self.run_inference(inputs_i)
            pass

        return val_loss

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
