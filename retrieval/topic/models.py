import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from indexing import TopicTermIndex, TopicQueryTermIndex, TermIndex


class TopicModel:
    log = logging.getLogger('TopicModel')

    def __init__(self, index: TermIndex):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        self.index = index

    def _score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the relevance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: relevance score
        """
        return 1.0

    def query(self, query: List[str], top_k: int = -1) -> pd.DataFrame:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: DataFrame with a column for topic score.
            Frame is sorted and reduced to top_k rows
        """
        self.log.debug('start topic process for query %s', query)
        if top_k < 0:
            top_k = len(self.index.get_document_ids())
        else:
            top_k = min(len(self.index.get_document_ids()), top_k)

        score = pd.DataFrame(index=self.index.get_document_ids(), columns=['topic'], dtype=float)
        for doc_id in score.index:
            score.loc[doc_id, 'topic'] = self._score(query, doc_id)

        return score.nlargest(top_k, 'topic', keep='all')


class DirichletLM(TopicModel):
    log = logging.getLogger('DirichletLM model')

    def __init__(self, index: TermIndex, alpha: int = 1000):
        """
        Constructor for a DirichletLM model.

        :param index: index to get relevance data from
        :param alpha: alpha parameter for Dirichlet smoothing
        """
        super().__init__(index)
        self.alpha = alpha

    def _score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the relevance of a document given a query using Dirichlet smoothing.

        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: relevance score
        """
        rho = 1
        doc_length = self.index.get_document_length(doc_id)
        total_doc_length = self.index.get_total_document_length()
        for term in query:
            frequency = self.index.get_term_frequency(term, doc_id)
            total_frequency = self.index.get_total_term_frequency(term)
            try:
                rho += np.log(self._term_probability(frequency, total_frequency, doc_length, total_doc_length))
            except ValueError:
                rho += 0
        return float(rho)

    def _term_probability(self, frequency: float, total_frequency: float, doc_length: float, total_doc_length: float):
        """
        Calculates the conditional probability of a term give a document using Dirichlet smoothing.
        :param frequency: term frequency of the current term and document
        :param total_frequency: total term frequency of the current term
        :param doc_length: length of the document to calculate the probability for
        :param total_doc_length: cumulative length of all documents in the collection
        """
        omega = self.alpha / (doc_length + self.alpha)
        if doc_length == 0:
            p1 = 0
        else:
            p1 = frequency / doc_length
        if total_doc_length == 0:
            p2 = 0
        else:
            p2 = total_frequency / total_doc_length
        return (1-omega) * p1 + omega * p2


class TopicRankingDirichlet(TopicModel):
    log = logging.getLogger('TopicModel')

    t_indexes: Dict[int, TopicTermIndex]
    tq_dirichlet: DirichletLM
    alpha: int

    def __init__(self, t_indexes: Dict[int, TopicTermIndex], tq_index: TopicQueryTermIndex,
                 alpha: int = 1000, tq_alpha: int = 1000):
        """
        Constructor for a TopicRankingDirichlet model.

        :param t_indexes: dict with topic indexes to get relevance data from
        :param tq_index: topic query term index to get topic data from
        :param alpha: alpha parameter for Dirichlet smoothing
        :param tq_alpha: alpha parameter for topic query Dirichlet smoothing
        """
        super().__init__(None)
        self.t_indexes = t_indexes
        self.alpha = alpha
        self.tq_dirichlet = DirichletLM(tq_index, alpha)

    def _score_topic(self, query: List[str]) -> int:
        """
        Calculates the relevance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :return: relevance score
        """
        return self.tq_dirichlet.query(query, top_k=1).index[0]

    def query(self, query: List[str], top_k: int = -1) -> pd.DataFrame:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: DataFrame with a column for topic score.
            Frame is sorted and reduced to top_k rows
        """
        self.log.debug('start topic process for query %s', query)
        topic = self._score_topic(query)

        return DirichletLM(self.t_indexes[topic], self.alpha).query(query, top_k)
