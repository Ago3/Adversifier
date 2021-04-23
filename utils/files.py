import pickle
from os import path, mkdir
from info import CACHE_DIR, LEXICON
from .correlation import logReg


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


def get_high_corr_words(dataset_name, data, class_id, cache=True):
    if cache:
        high_corr_words = load_from_cache('{}_{}_corr.pkl'.format(dataset_name, class_id))
    if not cache or not high_corr_words:
        high_corr_words = logReg(data, class_id)
        store_to_cache(high_corr_words, '{}_{}_corr.pkl'.format(dataset_name, class_id))
    return high_corr_words


def get_hateful_words():
    hateful_words = set()
    with open(LEXICON, 'r') as f:
        for line in f.readlines():
            hateful_words.add(line.split('\t')[4])
    return hateful_words


def read_tsv_datafile(file_name):
    with open(file_name, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        posts = [line[0] for line in lines]
        labels = [int(line[1]) for line in lines]
    return [posts, labels]
