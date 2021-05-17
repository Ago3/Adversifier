from info import LOG, RES_FILE
import os


def log(results):
	if not os.path.exists(LOG):
		os.mkdir(LOG)
	scores = {k : "%.2f" % (v * 100) for k, v in results.items() if not isinstance(v, list) }
	with open(RES_FILE, 'w+') as f:
		f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('f1_o', 'quoting_a_to_n', 'corr_n_to_n', 'flip_n_to_a', 'corr_a_to_a', 'aaa', 'hashtag_check'))
		f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(scores['f1_o'], scores['quoting_a_to_n'], scores['corr_n_to_n'], scores['flip_n_to_a'], scores['corr_a_to_a'], scores['aaa'], scores['hashtag_check']))
