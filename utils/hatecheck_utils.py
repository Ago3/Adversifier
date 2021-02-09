from info import HATECHECK_F20, HATECHECK_F21, HATECHECK_F18, HATECHECK_F19
import numpy as np


def get_counter_quote_instances():
    examples = list()
    with open(HATECHECK_F20, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_counter_ref_instances():
    examples = list()
    with open(HATECHECK_F21, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_ident_neutral_instances():
    examples = list()
    with open(HATECHECK_F18, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def get_ident_pos_instances():
    examples = list()
    with open(HATECHECK_F19, 'r') as f:
        for line in f.readlines():
            examples.append(line.split('\t')[1].strip())
    return examples


def evaluate_on_hatecheck_functionality(predictor, functionality):
    if functionality == 'f20':
        examples = get_counter_quote_instances()
    elif functionality == 'f21':
        examples = get_counter_ref_instances()
    elif functionality == 'f18':
        examples = get_counter_ref_instances()
    elif functionality == 'f19':
        examples = get_counter_ref_instances()
    else:
        print('The functionality is not supported')
    predictions = predictor([examples])
    labels = np.zeros((len(examples),))
    tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    print('TPR on HateCheck {}: '.format(functionality), tpr)


def evaluate_on_hatecheck(predictor):
    for functionality in ['f20', 'f21', 'f18', 'f19']:
        evaluate_on_hatecheck_functionality(predictor, functionality)

