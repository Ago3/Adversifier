from collections import Counter
import numpy as np
import string
from .twitter import preprocess_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def logReg(data, class_id):
    train_tweets, train_labels = data
    preprocessed_train_tweets = list()
    for post in train_tweets:
        post = ' '.join(w for w in post.split() if '#' not in w or class_id==0)
        preprocessed_train_tweets.append(preprocess_tweet(post))
    train_tweets = preprocessed_train_tweets
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1500) 
    train_data_features = vectorizer.fit_transform(train_tweets)
    print("Creating Features...")
    train_data_features = train_data_features.toarray()
    print("done! Going to Train the Data Features")
    print("The length of the array is " + str(len(train_data_features)))
    clf = LogisticRegression(random_state=0)
    print("Training..")
    clf = clf.fit(train_data_features, train_labels)
    print("Training Completed")
    features = feature_importance(clf.coef_, vectorizer.get_feature_names(), class_id)
    return features


def feature_importance(coef, names, class_id, top_k=100):
    imp = coef
    assert imp.shape[0] == 1, 'Labels are not binary'
    class_offset = [-1, 1][class_id]
    class_name = ['non_abusive', 'abusive'][class_id]
    values, c_names = zip(*sorted(zip(imp[0], names), reverse=True, key=lambda x: class_offset * x[0]))
    features = list(c_names[:100])
    return features
