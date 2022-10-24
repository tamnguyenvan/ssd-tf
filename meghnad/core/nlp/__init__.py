from meghnad.cfg.config import MeghnadConfig

config = MeghnadConfig()

import nltk

# Load common necessities
_ = nltk.download('stopwords', quiet=True, download_dir=config.get_meghnad_configs('NLTK_PATH'))
_ = nltk.download('wordnet', quiet=True, download_dir=config.get_meghnad_configs('NLTK_PATH'))
_ = nltk.download('punkt', quiet=True, download_dir=config.get_meghnad_configs('NLTK_PATH'))
_ = nltk.download('cmudict', quiet=True, download_dir=config.get_meghnad_configs('NLTK_PATH'))
_ = nltk.download('omw-1.4', quiet=True, download_dir=config.get_meghnad_configs('NLTK_PATH'))

