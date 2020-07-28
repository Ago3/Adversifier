from AAAdversifier import AAAdversifier
from utils import get_config
from info import DATASET
from random import randint


def toy_model(list_of_non_preprocessed_posts):
    list_of_predictions = [randint(0, 1) for p in list_of_non_preprocessed_posts]
    return list_of_predictions


def get_data():
    LABELS = ['neither', 'sexism', 'racism', 'both']
    with open(DATASET, 'r') as f:
        lines = f.readlines()
        posts = [line.split('\t')[1] for line in lines]
        labels = [LABELS.index(line.split('\t')[2].strip()) for line in lines]  # <--- Convert to 0 or 1
        labels = [l if l <= 1 else 1 for l in labels]
        return [posts, labels]


def main():
    # Toy example
    config = get_config()
    adversifier = AAAdversifier(config)
    adversifier.aaa(toy_model, get_data())  # Check arguments description in AAAdversifier.py


if __name__ == '__main__':
    main()
