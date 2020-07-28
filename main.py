from AAAdversifier import AAAdversifier
from utils import get_config
from info import DATASET, KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH
from random import randint
from models import KennedyModel


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
    # Example: Kennedy
    kennedy_model = KennedyModel(KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, 100)
    adversifier.aaa(kennedy_model.forward, get_data())


if __name__ == '__main__':
    main()
