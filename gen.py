from AAAdversifier import AAAdversifier
import argparse


def generate_datafiles(dataset_name, train='aaa_files/fake_input.tsv', test='aaa_files/fake_input_test.tsv', outdir='aaa_files2'):
    adversifier = AAAdversifier(dataset_name)
    adversifier.generate_aaa_datafiles(train, test, outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='default',
                        help='''String to identify the dataset''')
    parser.add_argument('--train', type=str, default=None,
                        help='''Name of the file containing the training set''')
    parser.add_argument('--test', type=str, default=None,
                        help='''Name of the file containing the test set''')
    args = parser.parse_args()
    generate_datafiles(args.dataset_name, 'input/' + args.train, 'input/' + args.test, 'input/aaa_files')
