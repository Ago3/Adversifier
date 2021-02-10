CONFIG_DIR = 'config'
CONFIG_FILE = '/'.join([CONFIG_DIR, 'example.json'])  # CUSTOMIZE
DATA = 'DATA'
CACHE_DIR = 'CACHE'
LEXICON = '/'.join([DATA, 'hurtlex_EN_1.2.tsv'])
TEMPLATES = '/'.join([DATA, 'templates.txt'])
LOG = 'LOG'
RES_FILE = LOG + '/results.tsv'


#Dataset files
TRAIN_DATASET = '/'.join([DATA, 'waseem_train_np.tsv'])  # CUSTOMIZE
TEST_DATASET = '/'.join([DATA, 'waseem_test_np.tsv'])  # CUSTOMIZE
#Davidson file
DAVIDSON_CSV_FILE = '/'.join([DATA, 'davidson_data.csv'])  # (download from https://github.com/t-davidson/hate-speech-and-offensive-language )
DAVIDSON_TRAIN_IDS = '/'.join([DATA, 'davidson_train_ids.csv'])
DAVIDSON_VAL_IDS = '/'.join([DATA, 'davidson_val_ids.csv'])
DAVIDSON_TEST_IDS = '/'.join([DATA, 'davidson_test_ids.csv'])


#Kennedy model parameters
KENNEDY_RACISM_MODEL_PATH = 'models/racism_new.v2.bin'
KENNEDY_SEXISM_MODEL_PATH = 'models/sexism_new.v2.bin'


#Mozafari model parameters
MOZAFARI_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_model.pt'


#Mozafari Biased model parameters
MOZAFARI_BIASED_MODEL_PATH = 'models/mozafari_32_2e-05_0.1_train_model.pt'


#Mozafari trained discarding hashtags
MOZAFARI_MODEL_NH_PATH = 'models/mozafari_32_2e-05_0.1_nh_model.pt'


#SVM parameters
SVM_SEXISM_MODEL_PATH = 'models/sexism_svm.pkl'
SVM_RACISM_MODEL_PATH = 'models/racism_svm.pkl'
SVM_SEXISM_VECTORIZER_PATH = 'models/sexism_vectorizer.pkl'
SVM_RACISM_VECTORIZER_PATH = 'models/racism_vectorizer.pkl'


#HateCheck files
HATECHECK_ROOT = DATA + '/hatecheck'
HATECHECK_F20 = HATECHECK_ROOT + '/counter_quote_nh.tsv'
HATECHECK_F21 = HATECHECK_ROOT + '/counter_ref_nh.tsv'
HATECHECK_F18 = HATECHECK_ROOT + '/ident_neutral_nh.tsv'
HATECHECK_F19 = HATECHECK_ROOT + '/ident_pos_nh.tsv'
