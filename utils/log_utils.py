from info import LOG, RES_FILE
import os


def log(model_name, results):
	if not os.path.exists(LOG):
		os.mkdir(LOG)
	scores = {k : v * 100 for k, v in results.items()}
	with open(RES_FILE, 'a+') as f:
		f.write('{}\t{0:.2f}\t{0:.2f}\t{0:.2f}\t{0:.2f}\t{0:.2f}\t{0:.2f}\t{0:.2f}\n'.format(model_name, scores['f1_o'], scores['quoting_a_to_n'], scores['corr_n_to_n'], scores['flip_n_to_a'], scores['corr_a_to_a'], scores['aaa'], scores['hashtag_check']))
