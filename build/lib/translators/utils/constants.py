class Constants:
    MAX_INPUT_LENGTH = 50

    # tokens
    SPECIAL_TOKEN_UNKNOWN = "<unk>"
    SPECIAL_TOKEN_PAD = "<pad>"
    SPECIAL_TOKEN_BOS = "<bos>"
    SPECIAL_TOKEN_EOS = "<eos>"

    # encoder
    ENCODER_STACK_SIZE = 4
    ENCODER_HIDDEN_SIZE = 1_000
    ENCODER_EMBEDDING_SIZE = 1_000

    # dataset
    DATASET_BASE_URL = (
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    )
    DATASET_DOWNLOADS_PATH = "./data/"
    DATASET_SPLITS = {
        "train": {"source": "train.de", "target": "train.en"},
        "val": {"source": "val.de", "target": "val.en"},
        "test": {"source": "test_2016_flickr.de", "target": "test_2016_flickr.en"},
    }
    pass
