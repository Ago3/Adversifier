CONFIG_DIR = 'config'
CONFIG_FILE = '/'.join([CONFIG_DIR, 'example.json'])  # CUSTOMIZE
DATA = 'DATA'
# ORG = '/'.join([DATA, 'org.tsv'])  # CUSTOMIZE
CACHE_DIR = 'CACHE'
LEXICON = '/'.join([DATA, 'hurtlex_EN_1.2.tsv'])


#Dataset files
TRAIN_DATASET = '/'.join([DATA, 'waseem_train_np.tsv'])  # CUSTOMIZE
TEST_DATASET = '/'.join([DATA, 'waseem_test_np.tsv'])  # CUSTOMIZE


#Kennedy model parameters
KENNEDY_RACISM_MODEL_PATH = 'models/racism.bin'
KENNEDY_SEXISM_MODEL_PATH = 'models/sexism.bin'


#Mozafari model parameters
MOZAFARI_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_model.pt'


#Mozafari Biased model parameters
MOZAFARI_BIASED_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_test_model.pt'
