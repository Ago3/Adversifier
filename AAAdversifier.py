from settings import create_setting, SETTING_NAMES
from utils import geometric_mean, setting_f1_score


class AAAdversifier():
    def __init__(self, config):
        self.group_indentifiers = config['group_indentifiers']
        self.dataset_name = config['dataset_name']
        self.scores = dict()

    def eval_setting(self, setting_name, model, data):
        print('\nSETTING: {}'.format(setting_name))
        setting = create_setting(setting_name)
        posts, labels = setting.run(params=[self.dataset_name, data, self.group_indentifiers])
        print('Generating predictions..')
        predictions = model(posts)
        self.scores[setting_name] = setting_f1_score(predictions, labels, setting_name)
        print('{} score: {}'.format(setting_name, self.scores[setting_name]))
        return self.scores[setting_name]

    def aaa(self, model, data):
        """Computes the model scores on the AAA benchmark.        
        
        Arguments:
            model {function} -- A function that takes as input a list of posts, and returns a list containing the corresponding predictions
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
