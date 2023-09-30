import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from translators.utils.constants import Constants
from translators.datasets.translation_dataset import TranslationDataset

from translators.networks.encoder import Encoder
from translators.networks.decoder import Decoder


# ignore warnings
warnings.filterwarnings("ignore")

# dataset
dataset = TranslationDataset("de_core_news_sm", "en_core_web_sm")
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, (source_vocab, target_vocab) = dataset.load("train", 256, current_device)

# encoder configs
enc_embedding_dim = 256
enc_lstm_layers = 4
enc_lstm_hidden_dim = 256
enc_bidirec = True
enc_dropout_p = 0.025

# decoder configs
dec_embedding_dim = 256

# init encoder
enc = Encoder(
    vocab_size=len(source_vocab.get_itos()),
    embedding_dim=enc_embedding_dim,
    lstm_num_layers=enc_lstm_layers,
    lstm_hidden_dim=enc_lstm_hidden_dim,
    lstm_bidirec=enc_bidirec,
    dropout_p=enc_dropout_p,
).to(current_device)

# init decoder
dec = Decoder(
    vocab_size=len(target_vocab.get_itos()),
    embedding_dim=dec_embedding_dim,
    enc_lstm_hidden_dim=enc_lstm_hidden_dim,
    enc_lstm_bidirec=enc_bidirec,
    enc_lstm_layers=enc_lstm_layers,
    bos_tok_id=source_vocab.get_stoi()[Constants.SPECIAL_TOKEN_BOS],
    use_stepwise_ctxt=True,
).to(current_device)

# training setup
torch.enable_grad()
train_loss_i = list()

# opt / lr
enc_opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
dec_opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

# training loop
for e in range(200):
    print(f"*** epoch {e} ***")
    for i, batch in enumerate(train_loader):
        b_source, b_target = batch

        lstm_hidden, lstm_ctxt, _ = enc(b_source)
        logits, _ = dec(
            lstm_hidden, lstm_ctxt, use_teacher_forcing=True, minibatch_target=b_target
        )

        log_probs = F.log_softmax(logits, dim=-1)

        # zero grads
        enc.zero_grad(set_to_none=True)
        dec.zero_grad(set_to_none=True)

        # loss
        loss = loss_function(log_probs.view(-1, log_probs.size(-1)), b_target.view(-1))

        # backprop
        loss.backward()
        enc_opt.step()
        dec_opt.step()

        # track
        train_loss_i.append(round(loss.item(), 2))

        if i % 5 == 0:
            print(np.mean(train_loss_i[-5:]))
            pass