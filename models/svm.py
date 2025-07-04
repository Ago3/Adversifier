from sklearn.svm import SVC
import numpy as np
from utils import preprocess_tweet
import pickle
from info import SVM_SEXISM_MODEL_PATH, SVM_RACISM_MODEL_PATH, SVM_SEXISM_VECTORIZER_PATH, SVM_RACISM_VECTORIZER_PATH, SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_HATE_SPEECH_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH


class SvmModel():
    # def __init__(self, dataset='waseem'):
    #     assert dataset in ['waseem', 'davidson'], 'Dataset {} is not supported'.format(dataset)
    #     if dataset == 'waseem':
    #         paths = [SVM_SEXISM_VECTORIZER_PATH, SVM_SEXISM_MODEL_PATH, SVM_RACISM_VECTORIZER_PATH, SVM_RACISM_MODEL_PATH]
    #     else:
    #         paths = [SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_HATE_SPEECH_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH]
    #     print('Loading first vectorizer from path ', paths[0])
    #     with open(paths[0], 'rb+') as f:
    #         self.first_vectorizer = pickle.load(f)
    #     print('Loading first SVM model from path ', paths[1])
    #     with open(paths[1], 'rb+') as f:
    #         self.first_model = pickle.load(f)
    #     print('Loading second vectorizer from path ', paths[2])
    #     with open(paths[2], 'rb+') as f:
    #         self.second_vectorizer = pickle.load(f)
    #     print('Loading second SVM model from path ', paths[3])
    #     with open(paths[3], 'rb+') as f:
    #         self.second_model = pickle.load(f)


    def __init__(self, dataset='waseem'):
        assert dataset in ['waseem', 'davidson'], 'Dataset {} is not supported'.format(dataset)
        self.dataset = dataset

        if dataset == 'waseem':
            self.paths = [SVM_SEXISM_VECTORIZER_PATH, SVM_SEXISM_MODEL_PATH, SVM_RACISM_VECTORIZER_PATH, SVM_RACISM_MODEL_PATH]
        else:
            self.paths = [SVM_HATE_SPEECH_VECTORIZER_PATH, SVM_HATE_SPEECH_MODEL_PATH, SVM_OFFENSIVE_VECTORIZER_PATH, SVM_OFFENSIVE_MODEL_PATH]


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


    def train_or_load(self, input_data=None, val_data=None, test_data=None, load=True, seed=42):
        """
        Loads models and vectorizers from disk if load=True.
        Otherwise, trains new models on input_data and saves them.
        """
        if load:
            print('Loading first vectorizer from path ', self.paths[0])
            with open(self.paths[0], 'rb') as f:
                self.first_vectorizer = pickle.load(f)
            print('Loading first SVM model from path ', self.paths[1])
            with open(self.paths[1], 'rb') as f:
                self.first_model = pickle.load(f)
            print('Loading second vectorizer from path ', self.paths[2])
            with open(self.paths[2], 'rb') as f:
                self.second_vectorizer = pickle.load(f)
            print('Loading second SVM model from path ', self.paths[3])
            with open(self.paths[3], 'rb') as f:
                self.second_model = pickle.load(f)
        else:
            if input_data is None:
                raise ValueError("Training data must be provided when load=False")

            print("Training models from scratch...")
            random.seed(seed)
            np.random.seed(seed)

            texts = [preprocess_tweet(t) for t in input_data[0]]
            val_texts = [preprocess_tweet(t) for t in val_data[0]]
            test_texts = [preprocess_tweet(t) for t in test_data[0]]
            y = np.array(input_data[1])

            # First model and vectorizer
            self.first_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1500)
            X1 = self.first_vectorizer.fit_transform(texts + val_texts + test_texts)
            self.first_model = SVC(gamma='auto', cache_size=12000, max_iter=-1, kernel='linear', probability=True, random_state=seed)
            self.first_model.fit(X1, y)

            # Save first model/vectorizer
            with open(self.paths[0], 'wb') as f:
                pickle.dump(self.first_vectorizer, f)
            with open(self.paths[1], 'wb') as f:
                pickle.dump(self.first_model, f)

            # Second model and vectorizer
            self.second_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1500)
            X2 = self.second_vectorizer.fit_transform(texts + val_texts + test_texts)
            self.second_model = SVC(gamma='auto', cache_size=12000, max_iter=-1, kernel='linear', probability=True, random_state=seed)
            self.second_model.fit(X2, y)

            # # Save second model/vectorizer
            # with open(self.paths[2], 'wb') as f:
            #     pickle.dump(self.second_vectorizer, f)
            # with open(self.paths[3], 'wb') as f:
            #     pickle.dump(self.second_model, f)

            print("Training complete.")
