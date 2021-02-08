CONFIG_DIR = 'config'
CONFIG_FILE = '/'.join([CONFIG_DIR, 'example.json'])  # CUSTOMIZE
DATA = 'DATA'
# ORG = '/'.join([DATA, 'org.tsv'])  # CUSTOMIZE
CACHE_DIR = 'CACHE'
LEXICON = '/'.join([DATA, 'hurtlex_EN_1.2.tsv'])
TEMPLATES = '/'.join([DATA, 'templates.txt'])
LOG = 'LOG'
RES_FILE = LOG + '/results.tsv'


#Dataset files
TRAIN_DATASET = '/'.join([DATA, 'waseem_train_np.tsv'])  # CUSTOMIZE
TEST_DATASET = '/'.join([DATA, 'waseem_test_np.tsv'])  # CUSTOMIZE


#Kennedy model parameters
KENNEDY_RACISM_MODEL_PATH = 'models/racism_new.v2.bin'
KENNEDY_SEXISM_MODEL_PATH = 'models/sexism_new.v2.bin'


#Mozafari model parameters
MOZAFARI_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_model.pt'


#Mozafari Biased model parameters
MOZAFARI_BIASED_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_train_model.pt'


#SVM parameters
SVM_SEXISM_MODEL_PATH = 'models/sexism_svm.pkl'
SVM_RACISM_MODEL_PATH = 'models/racism_svm.pkl'
SVM_SEXISM_VECTORIZER_PATH = 'models/sexism_vectorizer.pkl'
SVM_RACISM_VECTORIZER_PATH = 'models/racism_vectorizer.pkl'
