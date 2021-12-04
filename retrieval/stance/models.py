import logging
from typing import List, Tuple

from indexing import TermIndex


class StanceModel:
    log = logging.getLogger('StanceModel')

    def __init__(self, index: TermIndex):
        """
        Constructor for model base class,
        :param index: index to get relevance data from
        """
        self.index = index

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
        return argument_relevant, argument_relevant
