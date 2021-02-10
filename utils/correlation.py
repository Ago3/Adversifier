from collections import Counter
import numpy as np
import string
from .twitter import preprocess_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# from matplotlib import pyplot as plt

# def pmi(data, top_k=100, no_hashtag=False):
#     documents = ['', '']
#     high_corr_words = []
#     posts, labels = data
#     for post, label in zip(posts, labels):
#         # Here I'm discarding hashtags as well
#         # text = ' '.join(w.lower() for w in post.split() if all([c not in string.punctuation for c in w]))
#         post = ' '.join(w for w in post.split() if '#' not in w)
#         text = preprocess_tweet(post)
#         documents[label] += (' ' + text)
#     counters = []
#     total_counter = Counter([])
#     for document in documents:
#         counter = Counter(document.split())
#         counters.append(counter)
#         total_counter += counter
#     total = sum([v for k, v in total_counter.items()])
#     for label, counter in enumerate(counters):
#         pmi_scores = dict()
#         current_total = sum([v for k, v in counter.items()])
#         for k, v in counter.items():
#             p_x = total_counter[k] / total
#             p_x_given_y = v / current_total
#             pmi_scores[k] = np.log(p_x_given_y / p_x)
#         high_corr_words.append([k for k, v in sorted(pmi_scores.items(), reverse=True, key=lambda item: item[1])][:top_k])
#     return high_corr_words


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
    # for c, class_name in zip([-1, 1], ['non_abusive', 'abusive']):
    values, c_names = zip(*sorted(zip(imp[0], names), reverse=True, key=lambda x: class_offset * x[0]))
    features = list(c_names[:100])
    # Uncomment for plotting the features
    # values, c_names = values[:50], c_names[:50]
    # f = plt.figure() 
    # f.set_figwidth(7) 
    # f.set_figheight(7)
    # plt.barh(range(len(c_names)), values, align='center')
    # plt.yticks(range(len(c_names)), c_names)
    # plt.tight_layout()
    # plt.savefig('lr_features_{}.png'.format(class_name))
    return features

