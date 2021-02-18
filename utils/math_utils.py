import numpy as np
from sklearn.metrics import f1_score
# import settings
from scipy.stats import chisquare
# from sklearn.metrics import confusion_matrix


def geometric_mean(values, weights=None):
    if not weights:
        weights = [1] * len(values)
    return np.prod([pow(v, w) for v, w in zip(values, weights)])**(1/sum(weights))


def setting_score(predictions, labels, setting_name):
    if setting_name in ['f1_o', 'hashtag_check']:
        tpr = []
        for class_id in [0, 1]:
            c_predictions = [p for p, l in zip(predictions, labels) if l == class_id]
            c_labels = [class_id] * len(c_predictions)
            c_tp = (np.array(c_predictions) == np.array(c_labels)).sum()
            c_tpr = c_tp / len(c_predictions)
            tpr.append(c_tpr)
            print("Class: {} TPR: {}".format(class_id, c_tpr))
        return f1_score(predictions, labels, average='micro'), tpr[0], tpr[1]
    tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    return tpr



def is_significant(mean_score, new_score, eps=0.00001):
    # z = (new_score - mean_score) / std
    # p_value = scipy.stats.norm.sf(abs(z))*2
    mean_score += eps
    new_score += eps
    statistic = chisquare([new_score], [mean_score])[0]
    # print(chisquare([new_score], [mean_score]))
    critical_value = 3.84
    return statistic > critical_value