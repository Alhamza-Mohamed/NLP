import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords

def process_tweet(tweet):
    """
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)

    # remove hashtags
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and      # remove stopwords
            word not in string.punctuation):       # remove punctuation
            stem_word = stemmer.stem(word)         # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """
    Input:
        tweets: list of tweets
        ys: corresponding list of labels 0 or 1
    Output:
        freq: dictionary mapping each (word, label) pair to count
    """
    yslist = np.squeeze(ys).tolist()
    freq = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freq:
                freq[pair] += 1
            else:
                freq[pair] = 1

    return freq
