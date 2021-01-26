from collections import Counter
import numpy as np
import string
from .twitter import preprocess_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


def pmi(data, top_k=100, no_hashtag=False):
    documents = ['', '']
    high_corr_words = []
    posts, labels = data
    for post, label in zip(posts, labels):
        # Here I'm discarding hashtags as well
        # text = ' '.join(w.lower() for w in post.split() if all([c not in string.punctuation for c in w]))
        post = ' '.join(w for w in post.split() if '#' not in w)
        text = preprocess_tweet(post)
        documents[label] += (' ' + text)
    counters = []
    total_counter = Counter([])
    for document in documents:
        counter = Counter(document.split())
        counters.append(counter)
        total_counter += counter
    total = sum([v for k, v in total_counter.items()])
    for label, counter in enumerate(counters):
        pmi_scores = dict()
        current_total = sum([v for k, v in counter.items()])
        for k, v in counter.items():
            p_x = total_counter[k] / total
            p_x_given_y = v / current_total
            pmi_scores[k] = np.log(p_x_given_y / p_x)
        high_corr_words.append([k for k, v in sorted(pmi_scores.items(), reverse=True, key=lambda item: item[1])][:top_k])
    return high_corr_words


def logReg(data):
    train_tweets, train_labels = data
    print(set(train_labels))
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
    feature_importance(clf.coef_, vectorizer.get_feature_names())
    return clf, vectorizer


def feature_importance(coef, names):
    imp = coef
    print(imp.shape)
    for c in range(imp.shape[0]):
        values, c_names = zip(*sorted(zip(imp[c], names), reverse=True, key=lambda x: x[0]))
        values, c_names = values[:20], c_names[:20]
        plt.barh(range(len(c_names)), values, align='center')
        plt.yticks(range(len(c_names)), c_names)
        plt.show()
