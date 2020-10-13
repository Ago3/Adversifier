from settings import create_setting, SETTING_NAMES
from utils import geometric_mean, setting_f1_score


class AAAdversifier():
    def __init__(self, config):
        """Creates an istance of AAAdversifier.
                
        Arguments:
            config {dictionary} -- A dictionary contining the list of the group identifiers of interest and the dataset name (used as identifier). If the list of group identifiers is empty, a default one is used (check paper).
        """
        self.group_indentifiers = config['group_indentifiers']
        self.dataset_name = config['dataset_name']
        self.scores = dict()

    def eval_setting(self, setting_name, model, data):
        """Computes the model scores on the specified setting.        
        
        Arguments:
            setting_name {string} -- The name of the setting (e.g., one of the available attacks)
            model {function} -- A function that takes as input a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            data {list} -- List containing 2 lists in the form: [list_of_posts, list_of_labels]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
        
        Returns:
            float -- the score obtained by model under the specified setting
        """
        print('\nSETTING: {}'.format(setting_name))
        setting = create_setting(setting_name)
        posts, labels = setting.run(params=[self.dataset_name, data[:2], self.group_indentifiers])
        print('Generating predictions..')
        model_input = [posts] + ([] if len(data) == 2 else data[2:])
        predictions = model(model_input)
        self.scores[setting_name] = setting_f1_score(predictions, labels, setting_name)
        print('{} score: {}'.format(setting_name, self.scores[setting_name]))
        return self.scores[setting_name]

    def aaa(self, model, data):
        """Computes the model scores on the AAA benchmark.        
        
        Arguments:
            model {function} -- A function that takes as input a list of (NON-preprocessed) posts, and returns a list containing the corresponding predictions
            data {list} -- List containing 2 lists in the form: [list_of_posts, list_of_labels]. Each label should be in [0, 1], where 0 corresponds to the non-abusive class and 1 corresponds to the abusive class
        
        Returns:
            float -- the AAA score
        """
        print('\nRunning AAA benchmark')
        for setting in SETTING_NAMES:
            self.eval_setting(setting, model, data)
        non_abusive_score = geometric_mean([self.scores[k] for k in ['quoting_nr', 'id_e', 'pmi_n']])
        abusive_score = geometric_mean([self.scores[k] for k in ['pmi_a', 'quoting_a']])
        self.scores['aaa'] = geometric_mean([self.scores['f1_o'], self.scores['org_n'], non_abusive_score, abusive_score], [1, 0.5, 0.5, 1])
        print('AAA score: {}'.format(self.scores['aaa']))
        return self.scores['aaa']
