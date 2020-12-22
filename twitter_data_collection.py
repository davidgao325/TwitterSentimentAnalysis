# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:02:24 2020

@author: David
"""
from __future__ import print_function
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
import re
import dataset
from tweepy.streaming import StreamListener
from sqlalchemy.exc import ProgrammingError
import twitter_credentials

# Stream a dataset and store it in a database, and then convert it into csv
class StdOutListener(StreamListener):
    def on_status(self, status):
        print(status.text)
        if status.retweeted:
            return True
        id_str = status.id_str
        created = status.created_at
        text = status.text
        sentiment = TextBlob(text).sentiment
        polarity = sentiment.polarity
        fav = status.favorite_count
        retweets = status.retweet_count
        name = status.user.screen_name
        description = status.user.description
        loc = status.user.location
        user_created = status.user.created_at
        followers = status.user.followers_count
        table = db['Tweet Information']
        try:
            table.insert(dict(
            id_str=id_str,
            created=created,
            tweet = text,
            fav_count=fav,
            retweet_count = retweets,
            user_name=name,
            user_description=description,
            user_location=loc,
            user_created=user_created,
            user_followers=followers,
            polarity = polarity,
        ))
        except ProgrammingError as err:
            print(err)

    def on_error(self, status_code):
        if status_code == 420:
            return False
    def clean_tweet(self, tweet): 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                |(\w+:\/\/\S+)", " ", tweet).split())
    

if __name__ == '__main__':
    # Create connection to sqlite database
    db = dataset.connect("sqlite:///tweets.db")
    listener = StdOutListener()
    auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
    auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
    stream = Stream(auth, listener)

    # Search twitter for programming languages
    stream.filter(track=['trump', 'biden', 'america','american election'])
    
