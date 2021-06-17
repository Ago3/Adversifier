import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import chisquare, norm


def geometric_mean(values, weights=None):
    if not weights:
        weights = [1] * len(values)
    return np.prod([pow(v, w) for v, w in zip(values, weights)])**(1/sum(weights))


def setting_score(predictions, labels, setting_name):
    if setting_name in ['f1_o', 'hashtag_check']:
        for class_id in [0, 1]:
            c_predictions = [p for p, l in zip(predictions, labels) if l == class_id]
            c_labels = [class_id] * len(c_predictions)
            c_tp = (np.array(c_predictions) == np.array(c_labels)).sum()
            c_tpr = c_tp / len(c_predictions)
            print("Class: {} CPR: {}".format(class_id, c_tpr))
        return f1_score(predictions, labels, average='micro')
    tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    return tpr


def is_significant(expected_scores, observed_scores, p_value_ref=0.05, threshold=0.05):
    expected_scores *= 100
    observed_scores *= 100
    diff = abs(expected_scores - observed_scores) / expected_scores
    std = expected_scores * 0.2
    zscore = (observed_scores - expected_scores) / std
    p_value = norm.sf(abs(zscore)) * 2
    return p_value <= p_value_ref and diff >= threshold
