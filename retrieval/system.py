import logging

from indexing import Preprocessor, Index
from .models import Model

log = logging.getLogger('retrievalSystem')


class RetrievalSystem:

    def __init__(self, index: Index, model: Model):
        """
        Constructor
        :param index: index to get relevance data from
        :param model: retrieval model to calculate retrieval scores with
        """
        self.preprocessor = Preprocessor()
        self.index = index
        self.model = model

    def query(self, text, top_k=-1):
        """
        Queries a given text against the index using a Dirichlet smoothed language model
        :param text: query text
        :param top_k: number of top results to return
        :return: list of (doc_id, score) tuples descending by score for all documents in the vector space
        """
        log.debug('start retrieval for query "%s"', text)
        query = self.preprocessor.preprocess(text)
        scores = {}
        top_k = max(min(len(self.index.get_document_ids()), top_k), 0)
        for doc_id in self.index.get_document_ids():
            scores[doc_id] = self.model.score(query, doc_id)
        log.debug('scoring done, start sorting')
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
