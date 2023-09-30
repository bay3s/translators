import io
from collections import Counter
from typing import List, Tuple

import spacy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

from translators.utils.constants import Constants
from translators.utils.path_utils import dataset_filepaths


class TranslationDataset:
    def __init__(self, source_language: str, target_language: str):
        """
        Initialize the translation dataset.

        Args:
            source_language (str): The source language from which to translate text.
            target_language (str): The target language to which the text should be translated.
        """
        self.source_tokenizer = get_tokenizer("spacy", language=source_language)
        self.target_tokenizer = get_tokenizer("spacy", language=target_language)
        pass

    @staticmethod
    def _build_vocab(filepath: str, tokenizer: callable) -> spacy.Vocab:
        """
        Build vocabulary based on the provided filepath and tokenizer.

        Args:
                filepath (str): Filepath for the language data.
                tokenizer (spacy.tokenizer): Tokenizer associated with the language.

        Returns:
                spacy.Vocab
        """
        counter = Counter()

        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
                pass

        return vocab(
            counter,
            specials=[
                Constants.SPECIAL_TOKEN_UNKNOWN,
                Constants.SPECIAL_TOKEN_PAD,
                Constants.SPECIAL_TOKEN_BOS,
                Constants.SPECIAL_TOKEN_EOS,
            ],
        )

    def _build_dataset(
        self, source_filepath: str, target_filepath: str
    ) -> Tuple[List[Tuple], Tuple[spacy.Vocab, spacy.Vocab]]:
        """
        Construct the dataset given the source filepath and the target filepath.

        Args:
            source_filepath (str): The source file path.
            target_filepath (str): The target file path.

        Returns:
            Tuple[List[Tuple], Tuple[spacy.Vocab, spacy.Vocab]]
        """
        source_vocab = self._build_vocab(source_filepath, self.source_tokenizer)
        target_vocab = self._build_vocab(target_filepath, self.target_tokenizer)

        # raw data
        source_raw_iter = iter(io.open(source_filepath, encoding="utf8"))
        target_raw_iter = iter(io.open(target_filepath, encoding="utf8"))

        # build dataset (input, output) pairs
        dataset = list()

        for source_raw, target_raw in zip(source_raw_iter, target_raw_iter):
            source_tokenized = torch.tensor(
                [source_vocab[token] for token in self.source_tokenizer(source_raw)],
                dtype=torch.long,
            )

            target_tokenized = torch.tensor(
                [target_vocab[token] for token in self.target_tokenizer(target_raw)],
                dtype=torch.long,
            )

            dataset.append((source_tokenized, target_tokenized))
            pass

        return dataset, (source_vocab, target_vocab)

    @staticmethod
    def _collate_minibatch(
        minibatch: Tuple[List, List], source_vocab: spacy.Vocab, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate the minibatch.

        Args:
                minibatch (Tuple[List, List]): Tuple consisting of the source and target data.
                source_vocab (spacy.Vocab): Source vocabulary.
                device (torch.device): Device used for training / inference.

        Returns:
                Tuple[torch.Tensor, torch.Tensor]
        """
        # special tokens
        padding_token_id = source_vocab[Constants.SPECIAL_TOKEN_PAD]
        bos_token_id = source_vocab[Constants.SPECIAL_TOKEN_BOS]
        eos_token_id = source_vocab[Constants.SPECIAL_TOKEN_EOS]

        source_minibatch = list()
        target_minibatch = list()

        for source_item, target_item in minibatch:
            s = torch.cat(
                (
                    torch.tensor([bos_token_id]),
                    source_item,
                    torch.tensor([eos_token_id]),
                )
            )
            source_minibatch.append(s)

            t = torch.cat(
                (
                    torch.tensor([bos_token_id]),
                    target_item,
                    torch.tensor([eos_token_id]),
                )
            )
            target_minibatch.append(t)
            pass

        source_minibatch = pad_sequence(
            source_minibatch, padding_value=padding_token_id, batch_first=True
        )

        target_minibatch = pad_sequence(
            target_minibatch, padding_value=padding_token_id, batch_first=True
        )

        return source_minibatch.to(device), target_minibatch.to(device)

    def load(
        self, split_name: str, batch_size: int, device: torch.device
    ) -> Tuple[DataLoader, Tuple[spacy.Vocab, spacy.Vocab]]:
        """
        Returns a data loader that can be iterated over.

        Args:
                split_name (str): Dataset split.
                batch_size (int): Batch size.
                device (torch.device): Device used for training / inference.

        Returns:
                DataLoader
        """
        train_filepaths = dataset_filepaths(split_name)
        source_filepath = train_filepaths["source"]
        target_filepath = train_filepaths["target"]

        dataset, (source_vocab, target_vocab) = self._build_dataset(
            source_filepath, target_filepath
        )

        return DataLoader(
            dataset,
            collate_fn=lambda batch: self._collate_minibatch(
                batch, source_vocab, device
            ),
            batch_size=batch_size,
            shuffle=True,
        ), (source_vocab, target_vocab)
