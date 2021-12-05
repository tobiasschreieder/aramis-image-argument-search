import logging
from typing import List, Tuple

import pandas as pd

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

        topic_scores = self.topic_model.query(query)
        # nutze erste k elemente für argument
        argument_scores = self.argument_model.query(query, topic_scores)
        # nutze elemete die größer als 0 sind
        pro_scores, con_scores = self.stance_model.query(query, argument_scores)

        # TODO add normalised different scores together
        topic_df = pd.DataFrame(topic_scores, columns=['doc_id', 'topic']).set_index('doc_id')
        arg_df = pd.DataFrame(argument_scores, columns=['doc_id', 'arg']).set_index('doc_id')
        pro_df = pd.DataFrame(pro_scores, columns=['doc_id', 'pro']).set_index('doc_id')
        con_df = pd.DataFrame(con_scores, columns=['doc_id', 'con']).set_index('doc_id')

        df = pd.concat([topic_df, arg_df, pro_df, con_df], axis=1)
        df_norm = (df - df.min()) / (df.max() - df.min())

        t_w = 0.15
        a_w = 0.35
        s_w = 1 - t_w - a_w

        df_norm['p_score'] = t_w * df_norm['topic'] + a_w * df_norm['arg'] + s_w * df_norm['pro']
        df_norm['c_score'] = t_w * df_norm['topic'] + a_w * df_norm['arg'] + s_w * (1 - df_norm['con'])

        return self.get_sorted_list(df_norm['p_score'], top_k), self.get_sorted_list(df_norm['c_score'], top_k)

    @staticmethod
    def get_sorted_list(s: pd.Series, top_k: int) -> List[Tuple[str, float]]:
        result = []
        for e in s.dropna().iteritems():
            result.append(e)

        result = sorted(result, key=lambda item: item[1], reverse=True)

        if top_k < 0:
            top_k = len(result)
        else:
            top_k = min(len(result), top_k)

        return result[:top_k]
