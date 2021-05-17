from AAAdversifier import AAAdversifier
import argparse


def evaluate_answers():
    adversifier = AAAdversifier('waseem')
    adversifier.eval_aaa_answerfiles('aaa_answer_files')


def generate_datafiles():
    adversifier = AAAdversifier('waseem')
    adversifier.generate_aaa_datafiles('aaa_files/fake_input.tsv', 'aaa_files/fake_input_test.tsv', 'aaa_files2')


if __name__ == '__main__':
    # from os import listdir
    # from os.path import isfile, join
    # from random import randint
    # onlyfiles = [f for f in listdir('aaa_files') if isfile(join('aaa_files', f))]
    # for f in onlyfiles:
    #   with open(join('aaa_files', f), 'r') as f1, open(join('aaa_answer_files', f), 'w+') as f2:
    #       lines = [line.strip() for line in f1.readlines()]
    #       for line in lines:
    #           f2.write('{}\t{}\n'.format(line, randint(0, 1)))
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, default='gen',
                        help='''command to be run: 'gen' for generating data files, 'eval' for evaluating predictions''')
    args = parser.parse_args()
    if args.cmd == 'gen':
        generate_datafiles()
    elif args.cmd == 'eval':
        evaluate_answers()
    else:
        print('Invalid argument.')
