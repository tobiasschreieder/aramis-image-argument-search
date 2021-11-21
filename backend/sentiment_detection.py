# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:17:43 2021

@author: tobias
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer


#nltk.download('vader_lexicon')


test1 = "This is a stupid exampletext. I hate it"
test2 = "This is a cool exampletext."
test3 = "Expenses Total Income Elalemelaat deductions Income"

sid = SentimentIntensityAnalyzer()


def sentiment_nltk(sentence):
    sentiments = sid.polarity_scores(sentence)
    return sentiments['compound']
    

nltk1 = sentiment_nltk(test1)
nltk2 = sentiment_nltk(test2)
nltk3 = sentiment_nltk(test3)

