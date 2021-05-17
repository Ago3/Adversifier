from AAAdversifier import AAAdversifier
from utils import get_waseem_data


def main():
    adversifier = AAAdversifier('waseem')
    # data = get_waseem_data()
    # adversifier.generate_aaa_datafiles(data['train'], data['test'], 'aaa_files')
    adversifier.generate_aaa_datafiles('aaa_files/fake_input.tsv', 'aaa_files/fake_input_test.tsv', 'aaa_files2')


if __name__ == '__main__':
    main()
