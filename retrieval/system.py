import logging
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import minmax_scale

from indexing import Preprocessor
from .argument import ArgumentModel
from .stance import StanceModel
from .topic import TopicModel

log = logging.getLogger('retrievalSystem')


class RetrievalSystem:

    def __init__(self, prep: Preprocessor, topic_model: TopicModel,
                 argument_model: ArgumentModel, stance_model: StanceModel, topic_weight: float = 0.10):
        """
        Constructor
        :param topic_model: topic model to calculate topic scores with
        """
        self.prep = prep
        self.topic_model = topic_model
        self.argument_model = argument_model
        self.stance_model = stance_model
        self.topic_weight = topic_weight
        self.arg_weight = 1 - topic_weight

    def query(self, text: str, top_k: int = -1, **kwargs) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
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
        argument_scores = self.argument_model.query(query, topic_scores, **kwargs)
        pro_scores, con_scores = self.stance_model.query(query, argument_scores, **kwargs)

        if top_k < 0:
            top_k = len(topic_scores)

        pro = pd.DataFrame(minmax_scale(pro_scores), columns=pro_scores.columns, index=pro_scores.index)
        con = pd.DataFrame(minmax_scale(con_scores), columns=con_scores.columns, index=con_scores.index)
        con['stance'] = 1 - con['stance']

        ps = self.topic_weight * pro['topic'] + self.arg_weight * pro['argument']
        cs = self.topic_weight * con['topic'] + self.arg_weight * con['argument']

        # noinspection PyCompatibility
        return [e for e in ps.nlargest(top_k).iteritems()], [e for e in cs.nlargest(top_k).iteritems()]
