from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from utils import preprocess_tweet
import pickle
from info import SVM_SEXISM_MODEL_PATH, SVM_RACISM_MODEL_PATH, SVM_SEXISM_VECTORIZER_PATH, SVM_RACISM_VECTORIZER_PATH


class SvmModel():

    def __init__(self):
        print('Loading sexism vectorizer from path ', SVM_SEXISM_VECTORIZER_PATH)
        with open(SVM_SEXISM_VECTORIZER_PATH, 'rb+') as f:
            self.sexism_vectorizer = pickle.load(f)
        print('Loading sexism SVM model from path ', SVM_SEXISM_MODEL_PATH)
        with open(SVM_SEXISM_MODEL_PATH, 'rb+') as f:
            self.sexism_model = pickle.load(f)
        print('Loading racism vectorizer from path ', SVM_RACISM_VECTORIZER_PATH)
        with open(SVM_RACISM_VECTORIZER_PATH, 'rb+') as f:
            self.racism_vectorizer = pickle.load(f)
        print('Loading racism SVM model from path ', SVM_RACISM_MODEL_PATH)
        with open(SVM_RACISM_MODEL_PATH, 'rb+') as f:
            self.racism_model = pickle.load(f)


    def predictor(self, input_args):
        input_lines = input_args[0]  # this model only takes the posts as input
        input_lines = [preprocess_tweet(tweet) for tweet in input_lines]
        sexism_data_features = self.sexism_vectorizer.transform(input_lines)
        sexism_data_features = sexism_data_features.toarray()
        sexism_predictions = self.sexism_model.predict(sexism_data_features)
        racism_data_features = self.racism_vectorizer.transform(input_lines)
        racism_data_features = racism_data_features.toarray()
        racism_predictions = self.racism_model.predict(sexism_data_features)
        predictions = np.max([sexism_predictions, racism_predictions], axis=0)
        return predictions


    # def __train__(self, training_data):
    #     posts, labels = training_data
    #     train_data_features = self.vectorizer.fit_transform(posts)
    #     print("Creating Features...")
    #     train_data_features = train_data_features.toarray()
    #     print("done! Going to Train the Data Features")
    #     print("The length of the array is " + str(len(train_data_features)))
    #     print("Training..")
    #     self.model = self.model.fit(train_data_features, labels)
    #     print("Training Completed")
    #     with open(SVM_MODEL_PATH, 'wb+') as f:
    #         pickle.dump(self.model, f)
    #     print("Model saved to path ", SVM_MODEL_PATH)


# def svm_model(clean_train_tweets, clean_train_labels, clean_test_tweets, clean_test_labels, clf=None):
#     vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1500) 
#     train_data_features = vectorizer.fit_transform(clean_train_tweets)
#     print("Creating Features...")
#     train_data_features = train_data_features.toarray()
#     print("done! Going to Train the Data Features")
#     print("The length of the array is " + str(len(train_data_features)))
#     if not clf:
#         clf = SVC(gamma='auto', cache_size=12000, max_iter=-1, kernel='linear', probability=True)
#         print("Training..")
#         clf = clf.fit(train_data_features, clean_train_labels)
#         print("Training Completed")
#     else:
#         print('Model loaded!')
#     test_data_features = vectorizer.transform(clean_test_tweets)
#     print("Making the features from the Test set...")
#     test_data_features = test_data_features.toarray()
#     print("Going to predict the Test Features...")
#     result = clf.predict(test_data_features)
#     print("Prediction Completed!!")
#     f1, f1_3c = svm_evaluation(clean_test_labels, result)
#     feature_importance(clf.coef_, vectorizer.get_feature_names())
#     return clf, vectorizer, f1, f1_3c


# def svm_evaluation(clean_test_labels, predictions):
#     print('Instances: {}'.format(len(predictions)))
#     f1 = f1_score(clean_test_labels, predictions, average='micro')
#     print("4 classes\nF1 score on test set: {}".format(f1))
#     labels_3c = [l if l < 3 else 2 for l in clean_test_labels]
#     predictions_3c = [l if l < 3 else 2 for l in predictions]
#     f1_3c = f1_score(labels_3c, predictions_3c, average='micro')
#     print("3 classes\nF1 score on test set: {}".format(f1_3c))
#     f1_3c_w = f1_score(labels_3c, predictions_3c, average='weighted')
#     print("3 classes\nWeighted F1 score on test set: {}".format(f1_3c_w))
#     labels_3c = np.array(labels_3c)
#     predictions_3c = np.array(predictions_3c)
#     micro_per_class_f1 = ''
#     weighted_per_class_f1 = ''
#     for c in set(labels_3c):
#         c_labels = labels_3c[labels_3c == c]
#         c_predictions = predictions_3c[labels_3c == c]
#         f1 = f1_score(c_labels, c_predictions, average='micro')
#         print('Micro F1 score for class {}: {}'.format(LABELS[c], f1))
#         micro_per_class_f1 += str(f1) + '\t'
#         f1 = f1_score(c_labels, c_predictions, average='weighted')
#         print('Weighted F1 score for class {}: {}'.format(LABELS[c], f1))
#         weighted_per_class_f1 += str(f1) + '\t'
#     abusive_answers = None
#     abusive_labels = None
#     for c in [1, 2]:
#         if abusive_labels is None:
#             abusive_answers = predictions_3c[labels_3c==c]
#             abusive_labels = labels_3c[labels_3c==c]
#         else:
#             abusive_answers = np.concatenate([abusive_answers, predictions_3c[labels_3c==c]], axis=0)
#             abusive_labels = np.concatenate([abusive_labels, labels_3c[labels_3c==c]], axis=0)
#     abusive_f1_all = []
#     for average in ['micro', 'macro', 'weighted']:
#         f1 = f1_score(abusive_labels, abusive_answers, average=average)
#         print('{} F1 score for class {}: {}'.format(average, 'abusive', f1))
#         abusive_f1_all.append(f1)
#     print('SUMMARY: {}\t{}{}\t{}\t{}{}\n'.format(f1_3c, micro_per_class_f1, abusive_f1_all[0], f1_3c_w, weighted_per_class_f1, abusive_f1_all[2]))
#     return f1, f1_3c


    # def feature_importance(coef, names):
    #     imp = coef
    #     for c in range(imp.shape[0]):
    #         values, c_names = zip(*sorted(zip(imp[c], names), reverse=True, key=lambda x: abs(x[0])))
    #         values, c_names = values[:20], c_names[:20]
    #         plt.barh(range(len(c_names)), values, align='center')
    #         plt.yticks(range(len(c_names)), c_names)
    #         plt.show()

