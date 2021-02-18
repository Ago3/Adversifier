from sklearn.svm import SVC
import numpy as np
from utils import preprocess_tweet
import pickle
from info import SVM_SEXISM_MODEL_PATH, SVM_RACISM_MODEL_PATH, SVM_SEXISM_VECTORIZER_PATH, SVM_RACISM_VECTORIZER_PATH, SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_HATE_SPEECH_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH


class SvmModel():

    def __init__(self, dataset='waseem'):
        assert dataset in ['waseem', 'davidson'], 'Dataset {} is not supported'.format(dataset)
        if dataset == 'waseem':
            paths = [SVM_SEXISM_VECTORIZER_PATH, SVM_SEXISM_MODEL_PATH, SVM_RACISM_VECTORIZER_PATH, SVM_RACISM_MODEL_PATH]
        else:
            paths = [SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_HATE_SPEECH_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH]
        print('Loading first vectorizer from path ', paths[0])
        with open(paths[0], 'rb+') as f:
            self.first_vectorizer = pickle.load(f)
        print('Loading first SVM model from path ', paths[1])
        with open(paths[1], 'rb+') as f:
            self.first_model = pickle.load(f)
        print('Loading second vectorizer from path ', paths[2])
        with open(paths[2], 'rb+') as f:
            self.second_vectorizer = pickle.load(f)
        print('Loading second SVM model from path ', paths[3])
        with open(paths[3], 'rb+') as f:
            self.second_model = pickle.load(f)


    def predictor(self, input_args):
        input_lines = input_args[0]  # this model only takes the posts as input
        input_lines = [preprocess_tweet(tweet) for tweet in input_lines]
        c1_data_features = self.first_vectorizer.transform(input_lines)
        c1_data_features = c1_data_features.toarray()
        c1_predictions = self.first_model.predict(c1_data_features)
        c2_data_features = self.second_vectorizer.transform(input_lines)
        c2_data_features = c2_data_features.toarray()
        c2_predictions = self.second_model.predict(c2_data_features)
        predictions = np.max([c1_predictions, c2_predictions], axis=0)
        return predictions
