import logging
from typing import List, Tuple

from indexing import Preprocessor
from .argument import ArgumentModel
from .stance import StanceModel
from .topic import TopicModel

log = logging.getLogger('retrievalSystem')


class RetrievalSystem:

    def __init__(self, prep: Preprocessor, topic_model: TopicModel,
                 argument_model: ArgumentModel, stance_model: StanceModel,
                 topic_weight: float = 0.30, argument_weight: float = 0.40, prefetch_top_k: float = 2):
        """
        Constructor
        :param topic_model: topic model to calculate topic scores with
        """
        self.prep = prep
        self.topic_model = topic_model
        self.argument_model = argument_model
        self.stance_model = stance_model
        self.topic_weight = topic_weight
        self.arg_weight = argument_weight
        self.stance_weight = 1 - topic_weight - argument_weight
        self.prefetch_top_k = prefetch_top_k

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

        topic_scores = self.topic_model.query(query)
        argument_scores = self.argument_model.query(query, topic_scores)
        pro_scores, con_scores = self.stance_model.query(query, argument_scores)

        if top_k < 0:
            top_k = len(topic_scores)

        pro_scores = pro_scores.nlargest(int(self.prefetch_top_k*top_k), 'topic', keep='all')
        con_scores = con_scores.nlargest(int(self.prefetch_top_k*top_k), 'topic', keep='all')

        pro = (pro_scores - pro_scores.min()) / (pro_scores.max() - pro_scores.min())
        con = (con_scores - con_scores.min()) / (con_scores.max() - con_scores.min())
        con['stance'] = 1 - con['stance']

        ps = self.topic_weight * pro['topic'] + self.arg_weight * pro['argument'] + self.stance_weight * pro['stance']
        cs = self.topic_weight * con['topic'] + self.arg_weight * con['argument'] + self.stance_weight * con['stance']

        # noinspection PyCompatibility
        return [e for e in ps.nlargest(top_k).iteritems()], [e for e in cs.nlargest(top_k).iteritems()]
