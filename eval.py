from AAAdversifier import AAAdversifier
import argparse


def evaluate_answers(dataset_name, indir='aaa_answer_files'):
    adversifier = AAAdversifier(dataset_name)
    adversifier.eval_aaa_answerfiles(indir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='default',
                        help='''String to identify the dataset''')
    args = parser.parse_args()
    evaluate_answers(args.dataset_name, 'output/answer_files')
