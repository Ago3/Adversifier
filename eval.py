# docker build -t aaa .
# docker run -v ~/phd/research/aaa/Adversifier/aaa_answer_files:/aaa/output/answer_files aaa python3 eval.py --dataset_name waseem

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
