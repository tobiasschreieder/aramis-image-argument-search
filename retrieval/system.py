import logging
from typing import List, Tuple

from indexing import Index
from .argument import ArgumentModel
from .stance import StanceModel
from .topic import TopicModel

log = logging.getLogger('retrievalSystem')


class RetrievalSystem:

    def __init__(self, index: Index, topic_model: TopicModel,
                 argument_model: ArgumentModel, stance_model: StanceModel):
        """
        Constructor
        :param index: index to get relevance data from
        :param topic_model: topic model to calculate topic scores with
        """
        self.index = index
        self.topic_model = topic_model
        self.argument_model = argument_model
        self.stance_model = stance_model

    def query(self, text: str, top_k: int = -1) -> List[Tuple[str, float]]:
        """
        Queries a given text against the index
        :param text: query text
        :param top_k: number of top results to return
        :return: list of (doc_id, score) tuples descending by score for all documents in the vector space
        """
        log.debug('start retrieval for query "%s"', text)
        query = self.index.prep.preprocess(text)
        top_k = max(min(len(self.index.get_document_ids()), top_k), 0)

        topic_scores = self.topic_model.query(query, top_k)
        argument_scores = self.argument_model.query(query, topic_scores, top_k)
        stance_scores = self.stance_model.query(query, argument_scores, top_k)

        return stance_scores
