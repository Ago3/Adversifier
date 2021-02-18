import csv
from info import WASEEM_TSV_FILE, WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS


def get_waseem_data():
    id2post, id2label = __read_waseem_tsv_file__()
    files = [WASEEM_TRAIN_IDS, WASEEM_TEST_IDS]
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for filename, dataset_name in zip(files, ['train', 'test']):
        with open(filename, 'r') as f:
            lines = f.readlines()
            split_ids = [line.strip() for line in lines if line.strip() in id2post]
            split_posts = [id2post[idx] for idx in split_ids]
            split_labels = [LABELS.index(id2label[idx]) for idx in split_ids]
            if len(lines) > len(split_posts):
                print('Warning: {} tweets missing from {} set'.format(len(lines) - len(split_posts), dataset_name))
            assert len(split_posts) == len(split_labels), 'Posts and labels should be in same number'
            split_binary_labels = [l if l <= 1 else 1 for l in split_labels]
            extra_info_the_model_might_need = ['' for l in split_binary_labels]  # you can use this variable to pass, e.g., conversation context
            data[dataset_name] = [split_posts, split_binary_labels, extra_info_the_model_might_need]
    return data


def __read_waseem_tsv_file__():
    with open(WASEEM_TSV_FILE, 'r') as f:
        id2post = dict()
        id2label = dict()
        for line in f.readlines()[1:]:  # Skip header
            fields = line.split('\t')
            id_example = fields[0].strip()
            tweet = fields[1].strip()
            label = fields[2].strip()
            id2post[id_example] = tweet
            id2label[id_example] = label
        return id2post, id2label
