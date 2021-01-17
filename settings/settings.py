from abc import ABC, abstractmethod
from info import ORG
import random
from .strings import quoting_n_strings, group_identifiers, quoting_a_group_identifiers
from utils import get_high_corr_words, get_hateful_words


SETTING_NAMES = ['f1_o', 'hashtag_check', 'quoting_nr', 'pmi_n', 'pmi_a', 'quoting_a']
random.seed(1)


class Setting(ABC):
    @abstractmethod
    def run(self, params=None):
        pass

    def setup(self, params):
        pass


class Org_n(Setting):
    def run(self, params=None):
        with open(ORG, 'r') as f:
            posts = [line.split('\t')[1] for line in f.readlines()]
            labels = [0 for p in posts]
            return posts, labels


class Attack(Setting):
    def run(self, params=None):
        self.setup(params)
        posts, labels = params[1]
        new_posts, new_labels = list(), list()
        for post, label in zip(posts, labels):
            if label in self.affected_labels():
                new_posts.append(self.perturb_post(post))
                new_labels.append(self.get_label(label))
            # else:
            #     new_posts.append(post)
            #     new_labels.append(label)
        return new_posts, new_labels

    @abstractmethod
    def affected_labels(self):
        pass

    def get_label(self, label):
        pass

    def perturb_post(self, post):
        pass

    def setup(self, params):
        pass


class F1_o(Attack):
    def affected_labels(self):
        return []


class Quoting_nr(Attack):
    def perturb_post(self, post):
        return """{} {}""".format(post, random.choice(quoting_n_strings))

    def affected_labels(self):
        return [1]

    def get_label(self, label):
        return 0


class Id_e(Attack):
    def perturb_post(self, post):
        n = random.randint(1, 5)
        wordlist = [random.choice(self.group_identifiers) for i in range(n)]
        return "{} {}".format(post, ' '.join(wordlist))

    def affected_labels(self):
        return [0]

    def get_label(self, label):
        return label

    def setup(self, params):
        self.group_identifiers = params[-1] if params[-1] else group_identifiers


class Quoting_a(Attack):
    def perturb_post(self, post):
        self.identifier = random.choice(self.group_identifiers)
        return "{} only a {} could agree with this".format(post, self.identifier)

    def affected_labels(self):
        return [0]

    def get_label(self, label):
        return 1

    def setup(self, params):
        self.group_identifiers = params[-1] if params[-1] else quoting_a_group_identifiers


class PmiAttack(Attack):
    def perturb_post(self, post):
        n = random.randint(1, 10)
        wordlist = []
        for i in range(n):
            wordlist.append('#' + random.choice(self.relevant_words))  # e.g., Pmi_a: Pick n random words that are relevant for the non-abusive class
        return "{} {}".format(post, ' '.join(wordlist))

    def get_label(self, label):
        return label


class Pmi_a(PmiAttack):
    def affected_labels(self):
        return [1]

    def setup(self, params):
        self.relevant_words = get_high_corr_words(params[0], params[1], class_id=0)


class Pmi_n(PmiAttack):
    def affected_labels(self):
        return [0]

    def setup(self, params):
        hateful_words = get_hateful_words()
        high_corr_words = get_high_corr_words(params[0], params[1], class_id=1)
        self.relevant_words = list()
        for word in high_corr_words:
            if word not in hateful_words:
                self.relevant_words.append(word)
                if len(self.relevant_words) >= 50:
                    break


class Hashtag_check(Attack):
    @abstractmethod
    def affected_labels(self):
        return [0]

    def get_label(self, label):
        return 1

    def perturb_post(self, post):
        wordlist = ['#' + w for w in self.abusive_posts[self.idx]]
        self.idx = (self.idx + 1) % len(self.abusive_posts)
        return "{} {}".format(post, ' '.join(wordlist))

    def setup(self, params):
        self.idx = 0
        train_post, train_labels = params[-1]
        self.abusive_posts = list()
        for post, label in train_post, train_labels:
            if label == 1:
                self.abusive_posts.append(post)
                if len(self.abusive_posts) == len(params[1][0]):  #we need one abusive post for each training instance
                    break


def create_setting(setting_name):
    assert setting_name in SETTING_NAMES, 'The specified setting ({}) is not correct. Please select a setting from: {}'.format(setting_name, SETTING_NAMES)
    ss = [F1_o(), Hashtag_check(), Quoting_nr(), Pmi_n(), Pmi_a(), Quoting_a()]
    return ss[SETTING_NAMES.index(setting_name)]
