# TwitterSentimentAnalysis
A Twitter Sentiment Analysis is to use Natural Language Processing(NLP) to classify positive, neutral, and negative tweets to use for data analysis, text mining, data visualization. In this project, tweets based on the american election was extracted, and then later on analyzed and used to predict a sentiment score for each tweet.
```
TweetInformation.csv - Dataset created, upon streaming tweets into a database
updated_tweet_info.csv - Cleaned up dataset
twitter_eda.csv -Contains data from the analysis
```

# Data Collection
The data for this project was mainly extracted from tweepy and Textblob. Tweets were streamed and then created into a dataset, which contain columns such as, the text of the tweet, favorite count, retweet count, username, user description, user location, users followed, and the polarity. This data was then converted into a database, and the database was later converted into a csv file to use later on.

# Data Preprocessing and Cleaning
In this part of the project, tweets were being pre-processed and cleaned, so that it will be easier to predict the sentiment score of each tweet. Links, special characters, and  emojis were removed throughout this process. Stopwords, such as a and the were also removed, since they do not contribute to the overall meaning of a tweet. Polairty scores were also changed to -1,0, and 1 to make it easier to extract numerical features later on.

# Data Analysis
Word clouds were created to analyze the words used in the tweets. A general word cloud was created to see which words were commonly used. A negative word cloud was used to see which words contributed the most to a tweet resulting in a negative score. The same was created for a positive and neurtral word cloud. A bar graph was also created to examine the number of positive, negative, and neutral tweets there were.

# Model Building
Tweets were put into a testing and training dataset and then numerical features were extracted in tweets through the countvecotrizer. Tweets were later classified, using models such as, Naive Baynes, Logistic Regression, and DecisionTreeClassifier. A confusion matrix was used to examine the accuracy of the data prediction.
