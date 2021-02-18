DATA = 'DATA'
CACHE_DIR = 'CACHE'
LEXICON = '/'.join([DATA, 'hurtlex_EN_1.2.tsv'])
TEMPLATES = '/'.join([DATA, 'templates.txt'])
LOG = 'LOG'
RES_FILE = LOG + '/results.tsv'


#Waseem file
WASEEM_TSV_FILE = '/'.join([DATA, 'waseem_data.tsv'])  # put all tweets from Waseem et al., 2018 in this tsv file
WASEEM_TRAIN_IDS = '/'.join([DATA, 'waseem_train_ids.csv'])
WASEEM_VAL_IDS = '/'.join([DATA, 'waseem_val_ids.csv'])
WASEEM_TEST_IDS = '/'.join([DATA, 'waseem_test_ids.csv'])

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


#Mozafari trained discarding hashtags
MOZAFARI_MODEL_NH_PATH = 'models/mozafari_32_2e-05_0.1_nh_model.pt'


#SVM parameters
SVM_SEXISM_MODEL_PATH = 'models/sexism_svm.pkl'
SVM_RACISM_MODEL_PATH = 'models/racism_svm.pkl'
SVM_SEXISM_VECTORIZER_PATH = 'models/sexism_vectorizer.pkl'
SVM_RACISM_VECTORIZER_PATH = 'models/racism_vectorizer.pkl'


WASEEM_18_CHECKPOINTS = [SVM_SEXISM_MODEL_PATH, SVM_SEXISM_VECTORIZER_PATH, SVM_RACISM_MODEL_PATH, SVM_RACISM_VECTORIZER_PATH, MOZAFARI_MODEL_PATH, MOZAFARI_MODEL_NH_PATH, KENNEDY_SEXISM_MODEL_PATH, KENNEDY_RACISM_MODEL_PATH]
#Google Drive ids for each checkpoint
WASEEM_18_IDS = ['19uVCQm0o5IHOI3jlJ8EM8Bw1dAI1Fy91', '1LV5_KL-neQkm3sKGwjd3pIzPLk_yiP8h', '1vRqbuqXSUnqRK2ruL1a2WIDGVlcJh5QI', '1FS9vyHtjOUbSeXROt33RchDL13ZQsAeE', '1LyJAy74RzqGe2Hg-INZOjlXhEnDsTGWP', '1-tbY0IOzjvbcu2utZ4RF1biAiUpXiHpU', '1F0N0FZSBSkdm4EEGnH8mbB0m6FBDg4fj', '1TbWGI0142DpN4shmLctOlDlK0fY42-tU']


#Kennedy model trained on Davidson
KENNEDY_HATESPEECH_MODEL_PATH = 'models/davidson_hatespeech.bin'
KENNEDY_OFFENSIVE_MODEL_PATH = 'models/davidson_offensive.bin'


#Mozafari model trained on Davidson
MOZAFARI_DAVIDSON_MODEL_PATH = 'models/mozafari_16_2e-05_0.1_davidson_model.pt'


#SVM trained on Davidson
SVM_HATE_SPEECH_MODEL_PATH = 'models/hate_speech_svm.pkl'
SVM_OFFENSIVE_MODEL_PATH = 'models/offensive_svm.pkl'
SVM_HATE_SPEECH_VECTORIZER_PATH = 'models/hate_speech_vectorizer.pkl'
SVM_OFFENSIVE_VECTORIZER_PATH = 'models/offensive_vectorizer.pkl'


DAVIDSON_17_CHECKPOINTS = [SVM_HATE_SPEECH_MODEL_PATH, SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, MOZAFARI_DAVIDSON_MODEL_PATH, KENNEDY_HATESPEECH_MODEL_PATH, KENNEDY_OFFENSIVE_MODEL_PATH]
#Google Drive ids for each checkpoint
DAVIDSON_17_IDS = ['1MPpb-6TouSlkRJ0GkeYIwkG2R-UONZze', '1g9clFa9fENLjumFrTE7IMT849n5NmKjR', '15QvP5EGffUAwtkwSwfjJRrunnpwqdmNc', '1lqsNOTT7ZwIEgPClcrWeFN4j5WMeHbrr', '1FFspZaUiznGKpqBtaOTseqSF-ple5KOs', '17_AInLbhhx9M7I1ldFcrGOGxoNeXDAXa', '1JsamtJ8Xa27tG4yG_o6ufLSwTCKDcmTu']


#HateCheck files
HATECHECK_ROOT = DATA + '/hatecheck'
HATECHECK_F20 = HATECHECK_ROOT + '/counter_quote_nh.tsv'
HATECHECK_F21 = HATECHECK_ROOT + '/counter_ref_nh.tsv'
HATECHECK_F18 = HATECHECK_ROOT + '/ident_neutral_nh.tsv'
HATECHECK_F19 = HATECHECK_ROOT + '/ident_pos_nh.tsv'
