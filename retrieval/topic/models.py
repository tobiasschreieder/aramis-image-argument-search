import logging
from typing import List, Tuple

import numpy as np

from indexing import Index


class TopicModel:
    log = logging.getLogger('TopicModel')

    def __init__(self, index: Index):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        self.index = index

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the relevance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: relevance score
        """
        return 1.0

    def query(self, query: List[str], top_k: int = -1) -> List[Tuple[str, float]]:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: list of (doc_id, score) tuples descending by score for all documents in the vector space
        """
        self.log.debug('start topic process for query %s', query)
        scores = {}
        if top_k < 0:
            top_k = len(self.index.get_document_ids())
        else:
            top_k = min(len(self.index.get_document_ids()), top_k)
        for doc_id in self.index.get_document_ids():
            scores[doc_id] = self.score(query, doc_id)
        self.log.debug('scoring done, start sorting')
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


class DirichletLM(TopicModel):
    log = logging.getLogger('DirichletLM model')

    def __init__(self, index: Index, alpha: int = 1000):
        """
        Constructor for a DirichletLM model.

        :param index: index to get relevance data from
        :param alpha: alpha parameter for Dirichlet smoothing
        """
        super().__init__(index)
        self.alpha = alpha

    def score(self, query: List[str], doc_id: str) -> float:
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


class TopicRankingDirichlet(DirichletLM):
    log = logging.getLogger('TopicModel')

    def __init__(self, index: Index, alpha: int = 1000):
        """
        Constructor for a TopicRankingDirichlet model.

        :param index: index to get relevance data from
        :param alpha: alpha parameter for Dirichlet smoothing
        """
        super().__init__(index, alpha)

    def _score_topic(self, query: List[str], doc_id: str) -> float:
        """
        Calculates the relevance score for a document (given by index and doc_id) and query (give ans query term list)
        :param query: preprocessed query in list representation to calculate the relevance for
        :param doc_id: document to calculate the relevance for
        :return: relevance score
        """
        return 1.0

    def query(self, query: List[str], top_k: int = -1) -> List[Tuple[str, float]]:
        """
        Queries a given query against the index using a model scoring function

        :param query: preprocessed query in list representation to calculate the relevance for
        :param top_k: number of top results to return
        :return: list of (doc_id, score) tuples descending by score for all documents in the vector space
        """
        self.log.debug('start topic process for query %s', query)
        scores = {}
        top_k = max(min(len(self.index.get_document_ids()), top_k), 0)
        for doc_id in self.index.get_document_ids():
            scores[doc_id] = self.score(query, doc_id)
        self.log.debug('scoring done, start sorting')
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]