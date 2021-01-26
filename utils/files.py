import json
import pickle
from os import path, mkdir
from info import CONFIG_FILE, CACHE_DIR, LEXICON
from .correlation import pmi, logReg


def get_config():
    with open(CONFIG_FILE) as json_data_file:
        config = json.load(json_data_file)
    print('CONFIG:')
    for k, v in config.items():
        print('--> {}\t{}'.format(k, v))
    return config


def store_to_cache(structure, file_name):
    if not path.exists(CACHE_DIR):
        mkdir(CACHE_DIR)
    full_path = '/'.join([CACHE_DIR, file_name])
    with open(full_path, 'wb+') as f:
        pickle.dump(structure, f)


def load_from_cache(file_name):
    full_path = '/'.join([CACHE_DIR, file_name])
    if path.exists(full_path):
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def get_high_corr_words(dataset_name, data, class_id, cache=False):
    high_corr_words = load_from_cache(dataset_name + '_pmi.pkl')
    if not high_corr_words:
        logReg(data)
        exit(0)
        high_corr_words = pmi(data, no_hashtag=True)
        if cache:
            store_to_cache(high_corr_words, dataset_name + '_pmi.pkl')
    return high_corr_words[class_id]


def get_hateful_words():
    hateful_words = set()
    with open(LEXICON, 'r') as f:
        for line in f.readlines():
            hateful_words.add(line.split('\t')[4])
    return hateful_words
