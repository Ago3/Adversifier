import numpy as np
from sklearn.metrics import f1_score
import settings


def geometric_mean(values, weights=None):
    if not weights:
        weights = [1] * len(values)
    return np.prod([pow(v, w) for v, w in zip(values, weights)])**(1/sum(weights))


def setting_f1_score(predictions, labels, setting_name):
    if setting_name == 'f1_o':
        return f1_score(predictions, labels, average='weighted')
    if setting_name in settings.SETTING_NAMES[1:4] + [settings.SETTING_NAMES[-1]]:
        class_id = 0
    else:
        class_id = 1
    c_predictions = [p for p, l in zip(predictions, labels) if l == class_id]
    c_labels = [class_id] * len(c_predictions)
    return f1_score(c_predictions, c_labels, average='micro')
