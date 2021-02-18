from abc import ABC, abstractmethod
import random
import re
from utils import get_high_corr_words, get_hateful_words
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from info import TEMPLATES


SETTING_NAMES = ['f1_o', 'hashtag_check', 'quoting_a_to_n', 'corr_n_to_n', 'corr_a_to_a', 'flip_n_to_a']


class Setting(ABC):
    @abstractmethod
    def run(self, params=None):
        pass

    def setup(self, params):
        pass


class Attack(Setting):
    def run(self, params=None):
        self.setup(params)
        posts, labels = params[1]
        new_posts, new_labels = list(), list()
        for post, label in zip(posts, labels):
            if label in self.affected_labels():
                new_posts.append(self.perturb_post(post))
                new_labels.append(self.get_label(label))
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
        return [0, 1]

    def perturb_post(self, post):
        return post

    def get_label(self, label):
        return label


class Quoting_a_to_n(Attack):
    def perturb_post(self, post):
        template = random.choice(self.templates)
        new_post = template.replace("<tweet>", post)
        return new_post

    def affected_labels(self):
        return [1]

    def get_label(self, label):
        return 0

    def setup(self, params):
        with open(TEMPLATES, 'r') as f:
            self.templates = [line.strip() for line in f.readlines()]


class Flip_n_to_a(Attack):
    # From non-abusive to abusive
    def perturb_post(self, post):
        abusive_post = self.abusive_posts[self.idx]
        self.idx = (self.idx + 1) % len(self.abusive_posts)
        return "{} {}".format(abusive_post, post)  # We don't want the abusive post to be truncated, otherwise we can't control the final label

    def affected_labels(self):
        return [0]

    def get_label(self, label):
        return 1

    def setup(self, params):
        self.idx = 0
        test_post, test_labels = params[1]
        test_abusive_posts = [p for p, l in zip(test_post, test_labels) if l == 1]
        random.shuffle(test_abusive_posts)
        self.abusive_posts = test_abusive_posts


class CorrAttack(Attack):
    def perturb_post(self, post):
        n = random.randint(1, 5)
        wordlist = []
        for i in range(n):
            wordlist.append('#' + random.choice(self.relevant_words))  # e.g., Corr_a_to_a: Pick n random words that are relevant for the non-abusive class
        return "{} {}".format(post, ' '.join(wordlist))

    def get_label(self, label):
        return label


class Corr_a_to_a(CorrAttack):
    # From abusive to abusive
    def affected_labels(self):
        return [1]

    def setup(self, params):
        self.relevant_words = get_high_corr_words(params[0], params[-1], class_id=0)


class Corr_n_to_n(CorrAttack):
    # From non-abusive to non-abusive
    def affected_labels(self):
        return [0]

    def setup(self, params):
        wordnet_lemmatizer = WordNetLemmatizer()
        hateful_words = get_hateful_words()
        high_corr_words = get_high_corr_words(params[0], params[-1], class_id=1)
        self.relevant_words = list()
        for word in high_corr_words:
            if word not in hateful_words and wordnet_lemmatizer.lemmatize(word, pos='v') not in hateful_words:
                self.relevant_words.append(word)
                if len(self.relevant_words) >= 100:
                    break


class Hashtag_check(Attack):
    def affected_labels(self):
        return [0, 1]

    def get_label(self, label):
        return label

    def perturb_post(self, post):
        wordlist = ['#' + w for w in post.split() if '@' not in w]
        wordlist = [re.sub(r'##([^\s]+)', r'#\1', w) for w in wordlist]
        return ' '.join(wordlist)


def create_setting(setting_name):
    assert setting_name in SETTING_NAMES, 'The specified setting ({}) is not correct. Please select a setting from: {}'.format(setting_name, SETTING_NAMES)
    ss = [F1_o(), Hashtag_check(), Quoting_a_to_n(), Corr_n_to_n(), Corr_a_to_a(), Flip_n_to_a()]
    return ss[SETTING_NAMES.index(setting_name)]
