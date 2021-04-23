from settings import create_setting, SETTING_NAMES
from utils import geometric_mean, setting_score, get_high_corr_words, is_significant
import random
from utils import log
import os


class AAAdversifier():
    def __init__(self, dataset_name):
        """Creates an istance of AAAdversifier.
                
        Arguments:
            dataset_name {string} -- A string identifier of the dataset
        """
        print('Creating adversifier for the {} dataset'.format(dataset_name))
        self.dataset_name = dataset_name
        self.scores = dict()

    def eval_setting(self, setting_name, model, test_data, train_data):
        """Computes the model scores on the specified setting.        
        
        Arguments:
            setting_name {string} -- The name of the setting (e.g., one of the available attacks)
            model {function} -- A function that takes as input a list of arguments, the 1st one being a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            test_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
            train_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class

        Returns:
            float -- the score obtained by model under the specified setting
        """
        print('\nSETTING: {}'.format(setting_name))
        setting = create_setting(setting_name)
        posts, labels = setting.run(params=[self.dataset_name, test_data[:2], train_data[:2]])
        print('Generating predictions..')
        model_input = [posts] + ([] if len(test_data) == 2 else test_data[2:])
        predictions = model(model_input)
        if setting_name in ['f1_o', 'hashtag_check']:
            self.scores[setting_name], self.scores[setting_name + '_tnr'], self.scores[setting_name + '_tpr'] = setting_score(predictions, labels, setting_name)
        else:
            self.scores[setting_name] = setting_score(predictions, labels, setting_name)
        print('{} score: {}'.format(setting_name, self.scores[setting_name]))
        return self.scores[setting_name]

    def aaa(self, model_name, model, train_data, test_data):
        """Computes the model scores on the AAA benchmark. All scores are saved in info.RES_FILE .    
        
        Arguments:
            model_name {string} -- A string identifier of the model
            model {function} -- A function that takes as input a list of arguments, the 1st one being a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            train_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
            test_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class

        Returns:
            float -- the AAA score
        """
        print('\nRunning AAA evaluation')
        #Finding common words with high correlation with each class
        random.seed(0)
        self.scores = dict()
        for class_id in range(2):
            get_high_corr_words(self.dataset_name, train_data[:2], class_id=class_id, cache=False)
        for setting in SETTING_NAMES:
            #If hashtags are not being used by the model, just skip the corr_a_to_a and corr_n_to_n attacks
            if 'hashtag_check' in self.scores and (is_significant(self.scores['hashtag_check_tnr'], self.scores['f1_o_tnr']) or is_significant(self.scores['hashtag_check_tpr'], self.scores['f1_o_tpr']))  and 'corr' in setting:
                self.scores[setting] = 0
                continue
            self.eval_setting(setting, model, test_data, train_data)
        self.scores['aaa'] = geometric_mean([self.scores[k] for k in ['quoting_a_to_n', 'corr_n_to_n', 'corr_a_to_a', 'flip_n_to_a']])
        print('\nAAA score: {}'.format(self.scores['aaa']))
        log(model_name, self.scores)
        return self.scores['aaa']


    def generate_datafile(self, setting_name, test_data, train_data, outdir):
        print('\nSETTING: {}'.format(setting_name))
        setting = create_setting(setting_name)
        posts, labels = setting.run(params=[self.dataset_name, test_data[:2], train_data[:2]])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(os.path.join(outdir, '{}.tsv'.format(setting_name)), 'w+') as out:
            for post, label in zip(posts, labels):
                out.write('{}\t{}\n'.format(post, label))


    def generate_aaa_datafiles(self, test_data, train_data, outdir):
        for setting_name in SETTING_NAMES:
            self.generate_datafile(setting_name, test_data, train_data, outdir)


    def eval_answerfile(self, setting_name, indir):
        print('\nSETTING: {}'.format(setting_name))
        if not os.path.exists(os.path.join(indir, '{}.tsv'.format(setting_name))):
            print('Error: file {} doesn\'t exist'.format(os.path.join(indir, '{}.tsv'.format(setting_name))))
            exit(1)
        with open(os.path.join(indir, '{}.tsv'.format(setting_name)), 'r') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            # In HashtagCheck, tweets containing only a user tag are empty
            lines = [line if len(line) == 3 else [' '] + line for line in lines]
            posts = [line[0] for line in lines]
            labels = [line[1] for line in lines]
            predictions = [line[2] for line in lines]
        if setting_name in ['f1_o', 'hashtag_check']:
            self.scores[setting_name], self.scores[setting_name + '_tnr'], self.scores[setting_name + '_tpr'] = setting_score(predictions, labels, setting_name)
        else:
            self.scores[setting_name] = setting_score(predictions, labels, setting_name)
        print('{} score: {}'.format(setting_name, self.scores[setting_name]))
        return self.scores[setting_name]


    def eval_aaa_answerfiles(self, indir):
        self.scores = dict()
        for setting_name in SETTING_NAMES:
            self.eval_answerfile(setting_name, indir)
        self.scores['aaa'] = geometric_mean([self.scores[k] for k in ['quoting_a_to_n', 'corr_n_to_n', 'corr_a_to_a', 'flip_n_to_a']])
        print('\nAAA score: {}'.format(self.scores['aaa']))
        return self.scores['aaa']
