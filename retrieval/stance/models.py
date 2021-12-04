import logging
import math
from typing import List, Tuple

from indexing import FeatureIndex, ImageType


class StanceModel:
    log = logging.getLogger('StanceModel')

    def __init__(self, index: FeatureIndex):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        self.index: FeatureIndex = index

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the stance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: stance score
        """
        return 1.0

    def query(self, query: List[str], argument_relevant: List[Tuple[str, float]],
              top_k: int = -1) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: List of scored document ids
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: (list of pro (doc_id, score), list of con (doc_id, score))
         tuples descending by score for all documents in the vector space
        """
        self.log.debug('start stance process for query %s', query)
        pro_scores = {}
        con_scores = {}
        if top_k < 0:
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)
        for doc_id, arg_score in argument_relevant:
            score = self.score(query, doc_id)
            if score > 0:
                pro_scores[doc_id] = score
            elif score < 0:
                con_scores[doc_id] = score

        self.log.debug('scoring done, start sorting')
        return \
            sorted(pro_scores.items(), key=lambda item: item[1], reverse=True)[:top_k], \
            sorted(con_scores.items(), key=lambda item: item[1], reverse=False)[:top_k]


class StandardStanceModel(StanceModel):
    log = logging.getLogger('StandardStanceModel')

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the stance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: stance score
        """
        image_type = self.index.get_image_type(doc_id)

        percentage_green = self.index.get_image_percentage_green(doc_id)
        percentage_red = self.index.get_image_percentage_red(doc_id)
        percentage_bright = self.index.get_image_percentage_bright(doc_id)
        percentage_dark = self.index.get_image_percentage_dark(doc_id)
        image_average_color = self.index.get_image_average_color(doc_id)

        distance_to_green = math.sqrt(
            (image_average_color[0] - 0) ** 2 + (image_average_color[1] - 255) ** 2 + (image_average_color[2] - 0) ** 2)
        distance_to_red = math.sqrt(
            (image_average_color[0] - 255) ** 2 + (image_average_color[1] - 0) ** 2 + (image_average_color[2] - 0) ** 2)
        distance_to_black = math.sqrt(
            (image_average_color[0] - 0) ** 2 + (image_average_color[1] - 0) ** 2 + (image_average_color[2] - 0) ** 2)
        distance_to_white = math.sqrt(
            (image_average_color[0] - 255) ** 2 + (image_average_color[1] - 255) ** 2 + (
                    image_average_color[2] - 255) ** 2)

        # between 0 and 1 (above 30 ~1)
        color_mood = 0
        if image_type == ImageType.CLIPART:
            color_mood = (percentage_green * (100 / distance_to_green)) - (percentage_red * (100 / distance_to_red))
        elif image_type == ImageType.PHOTO:
            hue_factor = 0.2
            color_mood = ((percentage_green * (100 / distance_to_green)) -
                          (percentage_red * (100 / distance_to_red))) + hue_factor * \
                         ((percentage_bright * (100 / distance_to_white)) -
                          (percentage_dark * (100 / distance_to_black)))

        image_text_len = self.index.get_image_text_len(doc_id)
        # between 1 and 0 (above 80 ~0)
        len_words = 1 / (math.exp(0.04 * image_text_len))

        image_text_sentiment_score = self.index.get_image_text_sentiment_score(doc_id)

        score = (color_mood * len_words) + (image_text_sentiment_score * (1 - len_words))
        return score
