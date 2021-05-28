import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import chisquare, norm


def geometric_mean(values, weights=None):
    if not weights:
        weights = [1] * len(values)
    return np.prod([pow(v, w) for v, w in zip(values, weights)])**(1/sum(weights))


def setting_score(predictions, labels, setting_name):
    if setting_name in ['f1_o', 'hashtag_check']:
        # tpr = []
        cm = []
        for class_id in [0, 1]:
            c_predictions = [p for p, l in zip(predictions, labels) if l == class_id]
            c_labels = [class_id] * len(c_predictions)
            c_tp = (np.array(c_predictions) == np.array(c_labels)).sum()
            cm += [c_tp, len(c_predictions) - c_tp]
            c_tpr = c_tp / len(c_predictions)
            # tpr.append(c_tpr)
            print("Class: {} CPR: {}".format(class_id, c_tpr))
        return f1_score(predictions, labels, average='micro'), cm  # tpr[0], tpr[1]
    tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    return tpr


# def old_is_significant(mean_score, new_score, eps=0.00001):
#     # mean_score = 0.1
#     # new_score = 0.0
#     mean_score += eps
#     new_score += eps
#     statistic = chisquare([new_score], [mean_score])[0]
#     # print('Statistic: ', statistic)
#     critical_value = 3.84
#     return statistic > critical_value


def is_significant(expected_scores, observed_scores, p_value_ref=0.05, threshold=0.05):
    print(expected_scores, observed_scores)
    # expected_scores = [700, 500, 20000000, 30000000]
    # observed_scores = [680, 520, 20000001, 29999999]
    # chi, p_value = chisquare(observed_scores, f_exp=expected_scores, ddof=1)
    # diff = max([abs(e - o) / e for e, o in zip(expected_scores, observed_scores)])
    diff = abs(expected_scores - observed_scores) / expected_scores
    # print(chi, p_value, diff)
    std = expected_scores * 0.2
    zscore = (observed_scores - expected_scores) / std
    p_value = norm.sf(abs(z_score)) * 2
    print(p_value, diff)
    print(p_value <= p_value_ref, diff >= threshold)
    # expected_scores = [700, 500, 20, 30]
    # observed_scores = [680, 520, 21, 29]
    # chi, p_value = chisquare(observed_scores, f_exp=expected_scores)
    # diff = max([abs(e - o) / e for e, o in zip(expected_scores, observed_scores)])
    # print(chi, p_value, diff)
    # print(p_value <= p_value_ref, diff >= threshold)
    # expected_scores = [1900, 3000]
    # observed_scores = [2000, 2900]
    # chi, p_value = chisquare(observed_scores, f_exp=expected_scores)
    # diff = max([abs(e - o) / e for e, o in zip(expected_scores, observed_scores)])
    # print(chi, p_value, diff)
    # print(p_value <= p_value_ref, diff >= threshold)
    return p_value <= p_value_ref and diff >= threshold
