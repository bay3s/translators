import warnings
from translators.utils.trainer import Trainer
import torch

warnings.filterwarnings("ignore")

if __name__ == "__main__":
	current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trainer = Trainer(
		source_language="de_core_news_sm",
		target_language="en_core_web_sm",
		batch_size=32,
		enc_learning_rate=1e-4,
		dec_learning_rate=1e-4,
		num_epochs=200,
		device=current_device
	)

	trainer.train()
	pass
