from settings import create_setting, SETTING_NAMES
from utils import geometric_mean, setting_score, get_high_corr_words, is_significant
import random


class AAAdversifier():
    def __init__(self, config):
        """Creates an istance of AAAdversifier.
                
        Arguments:
            config {dictionary} -- A dictionary contining the list of the group identifiers of interest and the dataset name (used as identifier). If the list of group identifiers is empty, a default one is used (check paper).
        """
        self.group_indentifiers = None
        self.dataset_name = config['dataset_name']
        self.scores = dict()

    def eval_setting(self, setting_name, model, test_data, train_data):
        """Computes the model scores on the specified setting.        
        
        Arguments:
            setting_name {string} -- The name of the setting (e.g., one of the available attacks)
            model {function} -- A function that takes as input a list of arguments, the 1st one being a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            train_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
            test_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class

        Returns:
            float -- the score obtained by model under the specified setting
        """
        print('\nSETTING: {}'.format(setting_name))
        setting = create_setting(setting_name)
        posts, labels = setting.run(params=[self.dataset_name, test_data[:2], self.group_indentifiers, train_data[:2]])
        print('Generating predictions..')
        model_input = [posts] + ([] if len(test_data) == 2 else test_data[2:])
        predictions = model(model_input)
        if setting_name == 'f1_o':
            self.scores[setting_name] = setting_score(predictions, labels, setting_name)
        else:
            self.scores[setting_name] = setting_score(predictions, labels, setting_name)
        print('{} score: {}'.format(setting_name, self.scores[setting_name]))
        return self.scores[setting_name]

    def aaa(self, model, train_data, test_data):
        """Computes the model scores on the AAA benchmark.        
        
        Arguments:
            model {function} -- A function that takes as input a list of arguments, the 1st one being a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            train_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
            test_data {list} -- List in the form: [list_of_posts, list_of_labels, any_extra_info, your_model_might_need]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class

        Returns:
            float -- the AAA score
        """
        print('\nRunning AAA evaluation')
        #Finding non-rare words with high correlation with each class
        random.seed(0)
        self.scores = dict()
        for class_id in range(2):
            get_high_corr_words(self.dataset_name, train_data[:2], class_id=class_id, cache=False)
        for setting in SETTING_NAMES:
            #If hashtags are not being used by the model, just skip the corr_a_to_a and corr_n_to_n attacks
            if 'hashtag_check' in self.scores and is_significant(self.scores['hashtag_check'], self.scores['f1_o'])  and 'corr' in setting:
                self.scores[setting] = 0
                continue
            self.eval_setting(setting, model, test_data, train_data)
        # non_abusive_score = geometric_mean([self.scores[k] for k in ['quoting_nr', 'pmi_n']])
        # abusive_score = geometric_mean([self.scores[k] for k in ['pmi_a', 'quoting_a']])
        # self.scores['aaa'] = geometric_mean([self.scores['f1_o'], self.scores['org_n'], non_abusive_score, abusive_score], [1, 0.5, 0.5, 1])
        self.scores['aaa'] = geometric_mean([self.scores[k] for k in ['quoting_a_to_n', 'corr_n_to_n', 'corr_a_to_a', 'flip_n_to_a']])
        print('\nAAA score: {}'.format(self.scores['aaa']))
        return self.scores['aaa']
