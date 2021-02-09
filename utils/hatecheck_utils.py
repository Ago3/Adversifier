from info import HATECHECK_F20
import numpy as np


def get_counter_quote_instances():
	examples = list()
	with open(HATECHECK_F20, 'r') as f:
		for line in f.readlines():
			examples.append(line.split('\t')[1].strip())
	return examples


def evaluate_on_hatecheck_counter_quote(predictor):
	examples = get_counter_quote_instances()
	predictions = predictor([examples])
	labels = np.zeros((len(examples),))
	tp = (np.array(predictions) == np.array(labels)).sum()
    tpr = tp / len(predictions)
    print('TPR on HateCheck counter quote: ', tpr)
