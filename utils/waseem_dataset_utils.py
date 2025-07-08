import csv
from info import WASEEM_TSV_FILE, WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS


def get_waseem_data(include_validation=False):
    id2post, id2label = __read_waseem_tsv_file__()
    files = [WASEEM_TRAIN_IDS, WASEEM_TEST_IDS] if not include_validation else [WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS]
    splits = ['train', 'test'] if not include_validation else ['train', 'validation', 'test']
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for filename, dataset_name in zip(files, splits):
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


def get_disaggregated_waseem_data(include_validation=False):
    id2post, id2label = __read_waseem_tsv_file__()
    files = [WASEEM_TRAIN_IDS, WASEEM_TEST_IDS] if not include_validation else [WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS]
    splits = ['train', 'test'] if not include_validation else ['train', 'validation', 'test']
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for filename, dataset_name in zip(files, splits):
        with open(filename, 'r') as f:
            lines = f.readlines()
            split_ids = [line.strip() for line in lines if line.strip() in id2post]
            split_posts = [id2post[idx] for idx in split_ids]
            split_labels = [LABELS.index(id2label[idx]) for idx in split_ids]
            if len(lines) > len(split_posts):
                print('Warning: {} tweets missing from {} set'.format(len(lines) - len(split_posts), dataset_name))
            assert len(split_posts) == len(split_labels), 'Posts and labels should be in same number'
            split_sexism_labels = [1 if l in [1, 3] else 0 for l in split_labels]
            split_racism_labels = [1 if l in [2, 3] else 0 for l in split_labels]
            extra_info_the_model_might_need = ['' for l in split_labels]  # you can use this variable to pass, e.g., conversation context
            data[dataset_name] = [split_posts, split_sexism_labels, split_racism_labels, extra_info_the_model_might_need]
    return data



def create_waseem_huggingface_files():
    import json
    from utils import preprocess_tweet
    id2post, id2label = __read_waseem_tsv_file__()
    files = [WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS]
    splits = ['train', 'validation', 'test']
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for filename, dataset_name in zip(files, splits):
        with open(filename, 'r') as f:
            lines = f.readlines()
            split_ids = [line.strip() for line in lines if line.strip() in id2post]
            split_posts = [id2post[idx] for idx in split_ids]
            split_labels = [LABELS.index(id2label[idx]) for idx in split_ids]
            if len(lines) > len(split_posts):
                print('Warning: {} tweets missing from {} set'.format(len(lines) - len(split_posts), dataset_name))
            assert len(split_posts) == len(split_labels), 'Posts and labels should be in same number'
            split_binary_labels = [l if l <= 1 else 1 for l in split_labels]
            data[dataset_name] = [{"qid": i, "text": preprocess_tweet(split_posts[i]), "rule": "comparison" if split_binary_labels[i] else "nothate"} for i in range(len(split_ids))]
        with open(f"DATA/hf_talat/{dataset_name}.jsonl", "w+") as out:
            for instance in data[dataset_name]:
                json.dump(instance, out)
                out.write("\n")


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
