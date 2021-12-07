import logging
import math
from typing import List, Tuple

import pandas as pd

from indexing import FeatureIndex


class ArgumentModel:
    log = logging.getLogger('ArgumentModel')

    def __init__(self, index: FeatureIndex):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        self.index = index

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the argument score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: argument score
        """
        return 1.0

    def query(self, query: List[str], topic_relevant: pd.DataFrame,
              top_k: int = -1) -> List[Tuple[str, float]]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param topic_relevant: DataFrame with data for topic score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: given DataFrame with a additional column for argument score.
            Frame is sorted and reduced to top_k rows
        """
        self.log.debug('start argument process for query %s', query)

        if top_k < 0:
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)
        for doc_id in topic_relevant.index:
            topic_relevant.loc[doc_id, 'argument'] = self.score(query, doc_id)

        return topic_relevant.nlargest(top_k, 'argument', keep='all')


class StandardArgumentModel(ArgumentModel):
    log = logging.getLogger('StandardArgumentModel')

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the argument score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: argument score
        """
        image_roi_area = self.index.get_image_roi_area(doc_id)
        # use cazy function to get a score between 0 and 1 with optimum near 0.8
        diagramm_factor = self.log_normal_density_function(image_roi_area)

        image_text_sentiment_score = self.index.get_image_text_sentiment_score(doc_id)
        image_text_len = self.index.get_image_text_len(doc_id)
        # between 1 and 3 (above 80 ~3)
        len_words_value = 3 + (((-1) / (math.exp(0.04 * image_text_len))) * 2)
        text_sentiment_factor = len_words_value * abs(image_text_sentiment_score)

        # (number words - value) [0 - 0][40 - 1][110 - 2][asymptotisch 3]
        text_factor = (1 - (1 / (math.exp(0.01 * image_text_len)))) * 3

        html_sentiment_score = self.index.get_html_sentiment_score(doc_id)

        score = diagramm_factor + text_sentiment_factor + text_factor + html_sentiment_score
        return score

    @staticmethod
    def log_normal_density_function(x: float) -> float:
        if x == 0:
            return 0
        elif x == 1:
            return 0
        else:
            return ((1 / (math.sqrt(2 * math.pi) * 0.16 * (-x + 1))) * math.exp(
                ((math.log((-x + 1), 10) + 0.49) ** 2) / -0.0512) * 0.12)
