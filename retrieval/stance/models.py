import datetime
import logging
import math
from pathlib import Path
from typing import List, Tuple
from Bio import pairwise2

import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow import keras

from indexing import FeatureIndex, ImageType, features_NN_stance
from indexing.feature import sentiment_detection


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

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              top_k: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: Tuple of given DataFrame with a additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()
        if top_k < 0:
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)

        for doc_id in argument_relevant.index:
            score = self.score(query, doc_id)
            if score > 0:
                pro_scores.loc[doc_id, 'stance'] = score
            elif score < 0:
                con_scores.loc[doc_id, 'stance'] = score

        return pro_scores.nlargest(top_k, 'stance', keep='all'), con_scores.nlargest(top_k, 'stance', keep='all')


class StandardStanceModel(StanceModel):
    log = logging.getLogger('StandardStanceModel')

    def __init__(self, index: FeatureIndex, weights: List[float] = None):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        :param weights: weights for query
        """
        super().__init__(index)
        self.weights = weights

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
        html_sentiment_score = self.index.get_html_sentiment_score(doc_id)

        # between 0 and 3
        color_mood = 0
        if image_type == ImageType.CLIPART:
            color_mood = (percentage_green - percentage_red) * 3
        elif image_type == ImageType.PHOTO:
            hue_factor = 0.2
            # between -1 and 1
            color_mood = ((percentage_green / 100 - percentage_red / 100) * (1 - hue_factor)) + \
                         ((percentage_bright / 100 - percentage_dark / 100) * hue_factor)
            # between -3 and 3
            color_mood = color_mood * 3

        image_text_sentiment_score = self.index.get_text_sentiment_score(doc_id)
        image_text_len = self.index.get_text_len(doc_id)
        # between 1 and 3 (above 80 ~3)
        len_words_value = 3 + (((-1) / (math.exp(0.04 * image_text_len))) * 2)
        text_sentiment_factor = len_words_value * image_text_sentiment_score

        # shift between -3 and 3
        html_sentiment_score = html_sentiment_score * 3

        query_sentiment_score = sentiment_detection.sentiment_nltk(' '.join(query))
        query_negation = (query_sentiment_score < 0) if abs(query_sentiment_score) > 0.2 else False

        if query_negation:
            color_mood = color_mood * (-1)
            text_sentiment_factor = text_sentiment_factor * (-1)
            html_sentiment_score = html_sentiment_score * (-1)

        return color_mood, text_sentiment_factor, html_sentiment_score

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              top_k: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: Tuple of given DataFrame with a additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()
        if top_k < 0:
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)

        df = pd.DataFrame(index=argument_relevant.index, columns=['color_mood', 'image_text_sentiment_score',
                                                                  'html_sentiment_score'])

        for doc_id in argument_relevant.index:
            df.loc[doc_id, :] = self.score(query, doc_id)

        df_norm = df / df.abs().max()

        if self.weights is None:
            np_weights = np.array([1, 1, 1])
        else:
            np_weights = np.array(self.weights)

        np_weights = np_weights / np_weights.sum()

        for doc_id in argument_relevant.index:
            score = (df_norm.loc[doc_id, :].to_numpy() * np_weights).mean()
            argument_relevant.loc[doc_id, 'stance'] = score
            if score > 0:
                pro_scores.loc[doc_id, 'stance'] = score
            else:  # if score < 0:
                con_scores.loc[doc_id, 'stance'] = score

        return pro_scores.nlargest(top_k, 'stance', keep='all'), con_scores.nlargest(top_k, 'stance', keep='all')


class NNStanceModel(StanceModel):
    model: keras.Model

    def __init__(self, index: FeatureIndex, model_name: str):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        super().__init__(index)
        model_path = Path('indexing/models/' + str(model_name) + '/model.hS')
        if not model_path.exists():
            raise FileNotFoundError(f'The model {model_name} does not exists.')
        self.model = load_model(model_path.as_posix(), compile=False)

    @staticmethod
    def query_frequency(query: List[str], text: List[str]) -> float:
        rho = 1
        doc_length = len(text)
        if doc_length == 0:
            return 0
        count = 0
        for term in query:
            frequency = text.count(term)
            if frequency == 0:
                continue
            try:
                rho += math.exp(math.log(frequency / doc_length))
                count += 1
            except ValueError:
                rho += 0
        if not count == 0:
            return float(rho/count)
        else:
            return 0

    @staticmethod
    def context_sentiment(query: List[str], text: List[str]) -> float:
        val = 0
        scope = 5
        count = 0
        for term in query:
            last = 0
            try:
                while last < len(text):
                    last = text.index(term, last+scope)
                    context = text[last-scope:last+scope]
                    val += sentiment_detection.sentiment_nltk(' '.join(context))
                    count += 1
            except ValueError:
                pass
        if not count == 0:
            return val/count
        else:
            return 0

    @staticmethod
    def alignment_query(query: List[str], text: List[str]) -> float:
        # looking for exact alignments of query in text
        a = pairwise2.align.localxx(text, query)
        sum_score = 0
        number_alignments = 0
        for match in a:
            number_alignments += 1
            sum_score += match.score
        # normalize alignment-score
        if not number_alignments == 0:
            avg_score = sum_score / number_alignments
            return avg_score
        else:
            return 0

    def query(self, query: List[str], argument_relevant: pd.DataFrame,
              top_k: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries a given preprocessed query against the index using a model scoring function

        :param argument_relevant: DataFrame with data for topic and argument score
        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: Tuple of given DataFrame with a additional column for pro/con stance score.
            Frames are sorted and reduced to top_k rows
        """
        self.log.debug('start stance process for query %s', query)

        if top_k < 0:
            top_k = len(self.index)
        else:
            top_k = min(len(self.index), top_k)

        features_list = []

        stored_df = Path("data/feature_df_stance.csv")
        if stored_df.is_file():
            data = pd.read_csv("data/feature_df_stance.csv")
            for doc_id in argument_relevant.index:
                features_list.append(data.loc[data['image_id'] == doc_id].squeeze())
        else:
            print("no feature_index.csv found. Please check this!")
            then = datetime.datetime.now()
            with self.index:
                took_h1 = datetime.timedelta(seconds=0)
                took_h2 = datetime.timedelta(seconds=0)
                took_h3 = datetime.timedelta(seconds=0)
                took_i1 = datetime.timedelta(seconds=0)
                took_i2 = datetime.timedelta(seconds=0)
                took_i3 = datetime.timedelta(seconds=0)

                for doc_id in argument_relevant.index:
                    pd_series = self.index.get_all_features(doc_id)
                    pd_series['query_sentiment'] = " ".join(query)

                    pd_series['query_html_eq'] = self.query_frequency(query, self.index.get_html_text(doc_id))
                    pd_series['query_image_eq'] = self.query_frequency(query, self.index.get_image_text(doc_id))
                    pd_series['query_html_context'] = self.context_sentiment(query, self.index.get_html_text(doc_id))
                    pd_series['query_image_context'] = self.context_sentiment(query, self.index.get_image_text(doc_id))
                    pd_series['query_image_align'] = self.alignment_query(" ".join(query), " ".join(self.index.get_image_text(doc_id)))
                    features_list.append(pd_series)

                took_h1 /= len(argument_relevant.index)
                took_i1 /= len(argument_relevant.index)
                took_h2 /= len(argument_relevant.index)
                took_i2 /= len(argument_relevant.index)
                took_h3 /= len(argument_relevant.index)
                took_i3 /= len(argument_relevant.index)
                self.log.debug('HTML: eq: %s, context: %s, align: %s', took_h1, took_h2, took_h3)
                self.log.debug('Imag: eq: %s, context: %s, align: %s', took_i1, took_i2, took_i3)

                took = datetime.datetime.now() - then
                self.log.debug('Took %s', took)

        results = features_NN_stance.make_prediction(model=self.model, input_data=features_list)

        argument_relevant.loc[:, 'stance'] = np.nan
        pro_scores = argument_relevant.copy()
        con_scores = argument_relevant.copy()
        for i, doc_id in enumerate(argument_relevant.index):
            score = results[i]
            argument_relevant.loc[doc_id, 'stance'] = score
            if score > 0:
                pro_scores.loc[doc_id, 'stance'] = score
            elif score < 0:
                con_scores.loc[doc_id, 'stance'] = abs(score)

        return pro_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all'), \
            con_scores.dropna(axis=0).nlargest(top_k, 'stance', keep='all')
