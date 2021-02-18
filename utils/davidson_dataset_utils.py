import csv
from sklearn.model_selection import StratifiedShuffleSplit
import sys
sys.path.append(".")
from info import DAVIDSON_CSV_FILE, DAVIDSON_TRAIN_IDS, DAVIDSON_VAL_IDS, DAVIDSON_TEST_IDS
# from .twitter import preprocess_tweet


def get_davidson_data():
    ids, posts, labels = __read_davidson_csv_file__()
    files = [DAVIDSON_TRAIN_IDS, DAVIDSON_TEST_IDS]
    data = dict()
    for filename, dataset_name in zip(files, ['train', 'test']):
        with open(filename, 'r') as f:
            split_ids = [int(line.strip()) for line in f.readlines()]
            split_posts = [p for idx, p in enumerate(posts) if idx in split_ids]
            split_labels = [l for idx, l in enumerate(labels) if idx in split_ids]
            split_binary_labels = [0 if l == 2 else 1 for l in split_labels]
            extra_info_the_model_might_need = ['' for l in split_binary_labels]  # you can use this variable to pass, e.g., conversation context
            data[dataset_name] = [split_posts, split_binary_labels, extra_info_the_model_might_need]
    return data


def __read_davidson_csv_file__():
    with open(DAVIDSON_CSV_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        ids = list()
        posts = list()
        labels = list()
        # data = dict()
        for line in csvreader:
            id_example = line[0]
            tweet = line[6]
            label = int(line[5])
            ids.append(id_example)
            posts.append(tweet)
            labels.append(label)
        return ids, posts, labels


def __split_dataset__(ids, labels):
    X = ids
    Y = labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=1)
    for train_idx, not_train_idx in sss.split(X, Y):
        not_train_X = [X[i] for i in not_train_idx]
        not_train_Y = [Y[i] for i in not_train_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5, random_state=1)
        for next_val, next_test in sss2.split(not_train_X, not_train_Y):
            val_idx = [not_train_idx[i] for i in next_val]
            test_idx = [not_train_idx[i] for i in next_test]
    files = [DAVIDSON_TRAIN_IDS, DAVIDSON_VAL_IDS, DAVIDSON_TEST_IDS]
    datasets = [train_idx, val_idx, test_idx]
    for filename, dataset in zip(files, datasets):
        with open(filename, 'w+') as f:
            f.write('\n'.join([str(x) for x in dataset]))


# def __save_preprocessed_files__():
#     ids, posts, labels = __read_davidson_csv_file__()
#     files = [DAVIDSON_TRAIN_IDS, DAVIDSON_VAL_IDS, DAVIDSON_TEST_IDS]
#     outfiles = ['DATA/davidson_train_processed.tsv', 'DATA/davidson_val_processed.tsv', 'DATA/davidson_test_processed.tsv']
#     for filename, dataset_name, outfile in zip(files, ['train', 'val', 'test'], outfiles):
#         with open(filename, 'r') as f:
#             split_ids = [int(line.strip()) for line in f.readlines()]
#             split_posts = [p for idx, p in enumerate(posts) if idx in split_ids]
#             split_labels = [l for idx, l in enumerate(labels) if idx in split_ids]
#             split_binary_labels = [0 if l == 2 else 1 for l in split_labels]
#             extra_info_the_model_might_need = ['' for l in split_binary_labels]  # you can use this variable to pass, e.g., conversation context
#             # data[dataset_name] = [split_posts, split_binary_labels, extra_info_the_model_might_need]
#             with open(outfile, 'w+') as out:
#                 for i, post, label in zip(split_ids, split_posts, split_labels):
#                     out.write('{}\t{}\t{}\n'.format(i, preprocess_tweet(post.replace('\n', '  ')), label))


if __name__ == '__main__':
    ids, posts, labels = __read_davidson_csv_file__()
    __split_dataset__(ids, labels)
    __save_preprocessed_files__()
