import string
import re
from nltk.tokenize import TweetTokenizer


def preprocess_tweet(text):
    text = re.sub('\n', ' ', text)
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation or char in '@#']
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # convert text to lower-case
    nopunc = nopunc.lower()
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', nopunc)
    nopunc = re.sub(r'http\S+', '<url>', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '<user>', nopunc)
    # remove the # in #hashtag
    nopunc = re.sub(r'#([^\s]+)', r'<hashtag> \1 </hashtag>', nopunc)
    # remove repeated characters
    nopunc = ' '.join(tknzr.tokenize(nopunc))
    # remove numbers
    nopunc = re.sub('\d+', '<number>', nopunc)
    return nopunc
