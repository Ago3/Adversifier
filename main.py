from AAAdversifier import AAAdversifier
from utils import get_config, evaluate_on_hatecheck, get_davidson_data
from info import TRAIN_DATASET, TEST_DATASET, KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, MOZAFARI_MODEL_PATH, MOZAFARI_BIASED_MODEL_PATH, MOZAFARI_MODEL_NH_PATH
from random import randint
from models import KennedyModel, MozafariModel, SvmModel


def toy_model(list_of_arguments):
    list_of_non_preprocessed_posts = list_of_arguments[0]
    list_of_predictions = [randint(0, 1) for p in list_of_non_preprocessed_posts]
    return list_of_predictions


def get_waseem_data():
    LABELS = ['neither', 'sexism', 'racism', 'both']
    data = dict()
    for dataset, name in zip([TRAIN_DATASET, TEST_DATASET], ['train', 'test']):
        with open(dataset, 'r') as f:
            lines = f.readlines()
            posts = [line.split('\t')[1] for line in lines]
            labels = [LABELS.index(line.split('\t')[2].strip()) for line in lines]  # <--- Convert to 0 (not abusive) or 1 (abusive)
            labels = [l if l <= 1 else 1 for l in labels]
            extra_info_the_model_might_need = ['' for l in labels]  # you can use this variable to pass, e.g., conversation context
            data[name] = [posts, labels, extra_info_the_model_might_need]
    return data


def main():
    # Toy example
    print('Evaluating Random Classifier:')
    config = get_config()
    adversifier = AAAdversifier(config)
    data = get_waseem_data()
    adversifier.aaa('random', toy_model, data['train'], data['test'])  # Check arguments description in AAAdversifier.py
    
    # Example: Kennedy et al., 2020
    print('\nEvaluating Kennedy Classifier:')
    kennedy_model = KennedyModel(KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, 100)
    adversifier.aaa('kennedy', kennedy_model.forward, data['train'], data['test'])
    evaluate_on_hatecheck(kennedy_model.forward)
    
    # Example: Mozafari et al., 2019
    print('\nEvaluating Mozafari Classifier:')
    mozafari_model = MozafariModel(MOZAFARI_MODEL_PATH, 100)
    adversifier.aaa('mozafari', mozafari_model.forward, data['train'], data['test'])
    evaluate_on_hatecheck(mozafari_model.forward)

    # Example: Mozafari et al., 2019 overfitted
    print('\nEvaluating Mozafari Biased Classifier:')
    mozafari_biased_model = MozafariModel(MOZAFARI_BIASED_MODEL_PATH, 100)
    adversifier.aaa('mozafari-overfitted', mozafari_biased_model.forward, data['train'], data['test'])
    evaluate_on_hatecheck(mozafari_biased_model.forward)

    # Example: Mozafari et al., 2019 with a pre-processing that discards hashtags
    print('\nEvaluating Mozafari (no hashtags) Classifier:')
    mozafari_model_nh = MozafariModel(MOZAFARI_MODEL_NH_PATH, 100, use_hashtags=False)
    adversifier.aaa('mozafari-nh', mozafari_model_nh.forward, data['train'], data['test'])

    # Example: SVM
    print('\nEvaluating SVM Classifier:')
    svm_model = SvmModel()
    adversifier.aaa('svm', svm_model.predictor, data['train'], data['test'])
    evaluate_on_hatecheck(svm_model.predictor)

    davidson_data = get_davidson_data()
    adversifier.aaa('random', toy_model, davidson_data['train'], davidson_data['test'])


if __name__ == '__main__':
    main()
