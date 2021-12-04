import logging
from typing import List, Tuple

from indexing import Preprocessor
from .argument import ArgumentModel
from .stance import StanceModel
from .topic import TopicModel

log = logging.getLogger('retrievalSystem')


class RetrievalSystem:

    def __init__(self, prep: Preprocessor, topic_model: TopicModel,
                 argument_model: ArgumentModel, stance_model: StanceModel):
        """
        Constructor
        :param topic_model: topic model to calculate topic scores with
        """
        self.prep = prep
        self.topic_model = topic_model
        self.argument_model = argument_model
        self.stance_model = stance_model

    def query(self, text: str, top_k: int = -1) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Queries a given text against the index
        :param text: query text
        :param top_k: number of top results to return
        :return: (list of pro (doc_id, score), list of con (doc_id, score))
         tuples descending by score for all documents in the vector space
        """
        log.debug('start retrieval for query "%s"', text)
        query = self.prep.preprocess(text)

        topic_scores = self.topic_model.query(query, top_k)
        # nutze erste k elemente für argument
        argument_scores = self.argument_model.query(query, topic_scores, top_k)
        # nutze elemete die größer als 0 sind
        pro_scores, con_scores = self.stance_model.query(query, argument_scores, top_k)

        return pro_scores, con_scores
