from info import WASEEM_TSV_FILE, WASEEM_TRAIN_IDS, WASEEM_VAL_IDS, WASEEM_TEST_IDS


def get_waseem_data():
    id2post, id2label = __read_waseem_tsv_file__()
    files = [WASEEM_TRAIN_IDS, WASEEM_TEST_IDS]
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for filename, dataset_name in zip(files, ['train', 'test']):
        with open(filename, 'r') as f:
            split_ids = [line.strip() for line in f.readlines()]
            split_posts = [id2post[idx] if idx in id2post for idx in split_ids]
            split_labels = [LABELS.index(id2label[idx]) if idx in id2label for idx in split_ids]
            if len(split_ids) > len(split_posts):
                print('Warning: {} tweets missing from {} set'.format(len(split_ids) - len(split_posts), dataset_name))
            assert len(split_posts) == len(split_labels), 'Posts and labels should be in same number'
            split_binary_labels = [l if l <= 1 else 1 for l in split_labels]
            extra_info_the_model_might_need = ['' for l in split_binary_labels]  # you can use this variable to pass, e.g., conversation context
            data[dataset_name] = [split_posts, split_binary_labels, extra_info_the_model_might_need]
    return data


def __read_waseem_tsv_file__():
    with open(WASEEM_TSV_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        next(csvreader)  # Skip header
        id2post = dict()
        id2label = dict()
        # data = dict()
        for line in csvreader:
            id_example = line[0]
            tweet = line[1]
            label = line[2]
            id2post[id_example] = tweet
            id2label[id_example] = label
        return id2post, id2label


# def get_waseem_data():
#     data = dict()
#     for dataset, name in zip([TRAIN_DATASET, TEST_DATASET], ['train', 'test']):
#         with open(dataset, 'r') as f:
#             lines = f.readlines()
#             posts = [line.split('\t')[1] for line in lines]
#             labels = [LABELS.index(line.split('\t')[2].strip()) for line in lines]  # <--- Convert to 0 (not abusive) or 1 (abusive)
#             labels = [l if l <= 1 else 1 for l in labels]
#             extra_info_the_model_might_need = ['' for l in labels]  # you can use this variable to pass, e.g., conversation context
#             data[name] = [posts, labels, extra_info_the_model_might_need]
#     return data
