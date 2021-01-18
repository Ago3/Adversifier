from collections import Counter
import numpy as np
import string
from .twitter import preprocess_tweet


def pmi(data, top_k=50, no_hashtag=False):
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
