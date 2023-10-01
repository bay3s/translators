from translators.utils.constants import Constants


def remove_special_toks(string: str) -> str:
	"""
	Removes special tokens from a given string.

	Args:
		string (str): String from which the special characters should be removed.

	Returns:
		str
	"""
	for special_ch in [Constants.SPECIAL_TOKEN_BOS, Constants.SPECIAL_TOKEN_PAD, Constants.SPECIAL_TOKEN_EOS]:
		string = string.replace(special_ch, "")

	return string

