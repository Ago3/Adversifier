from AAAdversifier import AAAdversifier
from utils import get_config
from info import TRAIN_DATASET, TEST_DATASET, KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, MOZAFARI_MODEL_PATH, MOZAFARI_BIASED_MODEL_PATH
from random import randint
from models import KennedyModel, MozafariModel, SvmModel


def toy_model(list_of_arguments):
    list_of_non_preprocessed_posts = list_of_arguments[0]
    list_of_predictions = [randint(0, 1) for p in list_of_non_preprocessed_posts]
    return list_of_predictions


def get_data():
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for dataset, name in zip([TRAIN_DATASET, TEST_DATASET], ['train', 'test']):
        with open(dataset, 'r') as f:
            lines = f.readlines()
            posts = [line.split('\t')[1] for line in lines]
            labels = [LABELS.index(line.split('\t')[2].strip()) for line in lines]  # <--- Convert to 0 (not abusive) or 1 (abusive)
            labels = [l if l <= 1 else 1 for l in labels]
            extra_info_the_model_might_need = ['' for l in labels]  # you can use this variable to pass, e.g., conversation context or user-related info
            data[name] = [posts, labels, extra_info_the_model_might_need]
    return data


def main():
    # Toy example
    print('Evaluating Random Classifier:')
    config = get_config()
    adversifier = AAAdversifier(config)
    data = get_data()
    adversifier.aaa('random', toy_model, data['train'], data['test'])  # Check arguments description in AAAdversifier.py
    
    # Example: Kennedy et al., 2020
    print('\nEvaluating Kennedy Classifier:')
    kennedy_model = KennedyModel(KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, 100)
    adversifier.aaa('kennedy', kennedy_model.forward, data['train'], data['test'])
    
    # Example: Mozafari et al., 2019
    print('\nEvaluating Mozafari Classifier:')
    mozafari_model = MozafariModel(MOZAFARI_MODEL_PATH, 100)
    adversifier.aaa('mozafari', mozafari_model.forward, data['train'], data['test'])

    # Example: Mozafari et al., 2019 biased following Utama et al., 2020
    print('\nEvaluating Mozafari Biased Classifier:')
    mozafari_biased_model = MozafariModel(MOZAFARI_BIASED_MODEL_PATH, 100)
    adversifier.aaa('mozafari-overfitted', mozafari_biased_model.forward, data['train'], data['test'])

    # Example: SVM
    print('\nEvaluating SVM Classifier:')
    svm_model = SvmModel()
    adversifier.aaa('svm', svm_model.predictor, data['train'], data['test'])


if __name__ == '__main__':
    main()
