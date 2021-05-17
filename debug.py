from AAAdversifier import AAAdversifier
from utils import get_waseem_data, read_tsv_datafile
from info import KENNEDY_RACISM_MODEL_PATH, KENNEDY_SEXISM_MODEL_PATH, MOZAFARI_MODEL_PATH, MOZAFARI_MODEL_NH_PATH, MOZAFARI_DAVIDSON_MODEL_PATH, KENNEDY_HATESPEECH_MODEL_PATH, KENNEDY_OFFENSIVE_MODEL_PATH
from models import MozafariModel
from os import listdir, mkdir
from os.path import join


def generate_answers(predictor, input_files_dir, output_files_dir):
    files = [f for f in listdir(input_files_dir)]
    os.mkdir(output_files_dir)
    for f in files:
        data = read_tsv_datafile(join(input_files_dir, f))
        predictions = predictor(data[0])
        with open(join(output_files_dir, f), 'w+') as out:
            for d, p in zip(data, predictions):
                out.write('{}\t{}\t{}\n'.format(d[0], d[1], p))


def main():
    adversifier = AAAdversifier('waseem')
    adversifier.generate_aaa_datafiles('DATA/waseem_train_np_0.tsv', 'DATA/waseem_test_np_0.tsv', 'aaa_files')
    data = get_waseem_data()
    
    # Example: Mozafari et al., 2019
    print('\nEvaluating Mozafari Classifier:')
    mozafari_model = MozafariModel(MOZAFARI_MODEL_PATH, 100)
    aaa_score = adversifier.aaa('mozafari', mozafari_model.forward, data['train'], data['test'])
    generate_answers(mozafari_model.forward, 'aaa_files', 'aaa_answer_files')
    aaa_alternative_score = adversifier.eval_aaa_datafiles('aaa_answer_files')
    assert aaa_score == aaa_alternative_score, '{} != {}'.format(aaa_score, aaa_alternative_score)

    # Example: Mozafari et al., 2019 with a pre-processing that discards hashtags
    # print('\nEvaluating Mozafari (no hashtags) Classifier:')
    # mozafari_model_nh = MozafariModel(MOZAFARI_MODEL_NH_PATH, 100, use_hashtags=False)
    # adversifier.aaa('mozafari-nh', mozafari_model_nh.forward, data['train'], data['test'])


if __name__ == '__main__':
    main()
