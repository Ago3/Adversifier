from AAAdversifier import AAAdversifier
from utils import evaluate_on_hatecheck, get_davidson_data, get_waseem_data
from info import KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, MOZAFARI_MODEL_PATH, MOZAFARI_MODEL_NH_PATH, MOZAFARI_DAVIDSON_MODEL_PATH, KENNEDY_HATESPEECH_MODEL_PATH, KENNEDY_OFFENSIVE_MODEL_PATH
from random import randint
from models import KennedyModel, MozafariModel, SvmModel


def toy_model(list_of_arguments):
    list_of_non_preprocessed_posts = list_of_arguments[0]
    list_of_predictions = [randint(0, 1) for p in list_of_non_preprocessed_posts]
    return list_of_predictions


def main():
    # Toy example
    print('Evaluating Random Classifier:')
    adversifier = AAAdversifier('waseem')
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

    # Example: Mozafari et al., 2019 with a pre-processing that discards hashtags
    print('\nEvaluating Mozafari (no hashtags) Classifier:')
    mozafari_model_nh = MozafariModel(MOZAFARI_MODEL_NH_PATH, 100, use_hashtags=False)
    adversifier.aaa('mozafari-nh', mozafari_model_nh.forward, data['train'], data['test'])

    # Example: SVM
    print('\nEvaluating SVM Classifier:')
    svm_model = SvmModel()
    adversifier.aaa('svm', svm_model.predictor, data['train'], data['test'])
    evaluate_on_hatecheck(svm_model.predictor)

    adversifier = AAAdversifier('davidson')
    davidson_data = get_davidson_data()
    adversifier.aaa('random', toy_model, davidson_data['train'], davidson_data['test'])

    # Example: Kennedy et al., 2020
    print('\nEvaluating Kennedy Classifier on Davidson data:')
    kennedy_davidson_model = KennedyModel(KENNEDY_HATESPEECH_MODEL_PATH, KENNEDY_OFFENSIVE_MODEL_PATH, 100)
    adversifier.aaa('kennedy-davidson', kennedy_davidson_model.forward, davidson_data['train'], davidson_data['test'])

    # Example: Mozafari et al., 2019
    print('\nEvaluating Mozafari Classifier on Davidson data:')
    mozafari_davidson_model = MozafariModel(MOZAFARI_DAVIDSON_MODEL_PATH, 100)
    adversifier.aaa('mozafari-davidson', mozafari_davidson_model.forward, davidson_data['train'], davidson_data['test'])

    # Example: SVM
    print('\nEvaluating SVM Classifier on Davidson data:')
    svm_davidson_model = SvmModel(dataset='davidson')
    adversifier.aaa('svm-davidson', svm_davidson_model.predictor, davidson_data['train'], davidson_data['test'])


if __name__ == '__main__':
    main()
