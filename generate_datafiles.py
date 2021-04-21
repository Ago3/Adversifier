from AAAdversifier import AAAdversifier
from utils import get_waseem_data


def main():
    # Toy example
    print('Evaluating Random Classifier:')
    adversifier = AAAdversifier('waseem')
    data = get_waseem_data()
    adversifier.generate_aaa_datafiles(data['train'], data['test'], 'aaa_files')


if __name__ == '__main__':
    main()
