import re
import string
import numpy as np
import nltk
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

def confidence_ellipse(x, y, ax, n_std=3.0, edgecolor='red', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : 1D arrays
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse radius.
    edgecolor : str
        Color of the ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # Compute covariance matrix
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Ellipse radiuses
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Create ellipse
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor='none',
                      edgecolor=edgecolor,
                      **kwargs)

    # Scale by standard deviation
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    # Mean of the data
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    transf = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return ellipse