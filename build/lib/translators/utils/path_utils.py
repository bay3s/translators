from subprocess import check_output, CalledProcessError
from functools import lru_cache
import os

from translators.utils.constants import Constants


@lru_cache(maxsize=1)
def root() -> str:
    """
    Returns the absolute path of the repository root.

    Returns:
      str
    """
    try:
        base = check_output(["git", "rev-parse", "--show-toplevel"])
    except CalledProcessError:
        raise IOError("Current working directory is not a git repository")

    return base.decode("utf-8").strip()


def absolute_path(relative_path: str) -> str:
    """
    Returns the absolute path for a path given relative to the root of the git repository.

    Args:
      relative_path (str): Relative path of the file / folder in the repo.

    Returns:
        str
    """
    return os.path.join(root(), relative_path)


def downloads_path() -> str:
    """
    Returns the absolute path for downloads associated with the datasets.

    Returns:
        str
    """
    return absolute_path("./data/")


def dataset_filepaths(dataset_split: str):
    """
    Get the source and target file paths associated with a specific dataset split.

    Args:
        dataset_split (str): Whether to return filepaths associated with the "train", "val", or "test" set.

    Returns:
        dict
    """
    if dataset_split not in ["train", "val", "test"]:
        return ValueError

    # files
    source_file = Constants.DATASET_SPLITS[dataset_split]["source"]
    target_file = Constants.DATASET_SPLITS[dataset_split]["target"]

    return {
        "source": f"{downloads_path()}{dataset_split}/{source_file}",
        "target": f"{downloads_path()}{dataset_split}/{target_file}",
    }
