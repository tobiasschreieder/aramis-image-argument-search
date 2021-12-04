# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:17:43 2021

@author: tobias
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


# calculate sentiment-score for given string
def sentiment_nltk(sentence):
    sentiments = sid.polarity_scores(sentence)
    return sentiments['compound']
