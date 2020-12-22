# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:15:07 2020

@author: David
"""
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from textblob import TextBlob
import re
import pandas as pd

# Read the file and create store the stopwords
df = pd.read_csv("Tweet Information.csv")
cache_english_stopwords = nltk.corpus.stopwords.words('english')

# Remove all punctuation, whitepsace, links, and stop words
def clean_tweet(tweet):
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_punctuation)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words)
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ')
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if c <= '\uFFFF')
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tw_list = tknzr.tokenize(tweet_no_emojis)
    list_no_stopwords = [i for i in tw_list if i not in cache_english_stopwords]
    tweet_filtered =' '.join(list_no_stopwords)
    return tweet_filtered

# Create new columns for clean tweets and sentiment scores
df['tweet'] = df['tweet'].apply(lambda x: clean_tweet(x))


po = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
df['new_polarity'] = df['tweet'].apply(po)
df['subjectivity'] = df['tweet'].apply(sub)

# Classify polairty score and import it into a new dataset
def updated_polarity(nums):
        if nums > 0:
            return 1
        if nums< 0:
            return -1
        else:
            return 0
df['clean_polarity'] = df['polarity'].apply(lambda x: updated_polarity(x))


updated_dataset = df.to_csv("updated_tweet_info.csv")

    



    
    



