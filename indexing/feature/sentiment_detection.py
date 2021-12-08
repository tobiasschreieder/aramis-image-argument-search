from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


def sentiment_nltk(sentence: str) -> float:
    """
    Calculate sentiment-score with Vader-Lexicon for given string
    :param sentence: String with text which should be analyzed
    :return: Calculated score as float
    """
    sentiments = sid.polarity_scores(sentence)
    return sentiments['compound']
