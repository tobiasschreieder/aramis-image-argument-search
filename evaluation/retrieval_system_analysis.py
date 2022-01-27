import datetime
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic
from retrieval import RetrievalSystem, TopicRankingDirichlet, NNArgumentModel, NNStanceModel
from .analysis_helper import get_relevant_eval, calc_precision_recall
from .configuration import Configuration

log = logging.getLogger('rs_analysis')

tq_index: TopicQueryTermIndex = None
topic_indexes: Dict[int, TopicTermIndex] = None
findex: FeatureIndex = None
loaded = False


def load_indices() -> None:
    global tq_index, topic_indexes, findex, loaded
    if loaded:
        return
    tq_index = TopicQueryTermIndex.load()
    topic_indexes = get_all_topic_indexes()
    findex = FeatureIndex.load(23158)
    loaded = True


def get_retrieval_system(cfg: Configuration, arg_model: str, stance_model: str) -> RetrievalSystem:
    load_indices()
    return RetrievalSystem(
        tq_index.prep,
        topic_model=TopicRankingDirichlet(
            t_indexes=topic_indexes, tq_index=tq_index, alpha=cfg.alpha, tq_alpha=cfg.tq_alpha
        ),
        argument_model=NNArgumentModel(findex, arg_model),
        stance_model=NNStanceModel(findex, stance_model),
        topic_weight=cfg.topic_weight,
        argument_weight=cfg.argument_weight,
        prefetch_top_k=cfg.prefetch_top_k,
    )


def plot_topic_precision_recall_at_k(rs: RetrievalSystem, topics: List[int], max_k: int = 50) -> None:
    topics = [Topic.get(topic) for topic in topics]
    df = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=['topic', 'k']),
                      columns=['pro_precision', 'pro_recall', 'con_precision', 'con_recall', 'precision', 'recall'])
    for topic in topics:
        relevant_p, relevant_c = get_relevant_eval(topic)
        result_p, result_c = rs.query(topic.title, top_k=max_k)
        result_p = [s[0] for s in result_p]
        result_c = [s[0] for s in result_c]
        for k in range(1, max_k, 2):
            pp, pr = calc_precision_recall(result_p[0:k], relevant_p)
            cp, cr = calc_precision_recall(result_c[0:k], relevant_c)
            p = (pp + cp) / 2
            r = (pr + cr) / 2
            df.loc[(topic.number, k), :] = [pp, pr, cp, cr, p, r]
    pass


def avg_topic_precision(rs: RetrievalSystem, topics: List[Topic], k) -> Tuple[float, float, float]:
    pro_error = 0
    con_error = 0
    for topic in topics:
        relevant_p, relevant_c = get_relevant_eval(topic)
        result_p, result_c = rs.query(topic.title, top_k=k)
        result_p = [s[0] for s in result_p]
        result_c = [s[0] for s in result_c]
        pp, _ = calc_precision_recall(result_p, relevant_p)
        cp, _ = calc_precision_recall(result_c, relevant_c)
        pro_error += pp
        con_error += cp
    pro_error /= len(topics)
    con_error /= len(topics)
    return pro_error, con_error, (pro_error + con_error) / 2


def find_rs_weights(rs: RetrievalSystem, topics: List[Topic], optimum: str = 'both', k: int = 20,
                    eta: int = 5) -> Tuple[float, float, float]:
    if optimum.lower() == 'pro':
        precision_pos = 0
    elif optimum.lower() == 'con':
        precision_pos = 1
    else:
        precision_pos = 2

    logging.getLogger().setLevel(logging.INFO)

    # best_precision = 0.1895
    # best_weights = (0.33, 0.66, 1 - 0.33 - 0.66)
    best_precision = 0
    best_weights = (0.5, 0.5, 0)
    i = 5
    for w_top in np.linspace(0, 1, 5):
        w_arg = 1 - w_top
        i -= 1
        log.info(f'cur w:%s', (w_top, w_arg, 1 - w_top - w_arg))
        then = datetime.datetime.now()
        rs.topic_weight = w_top
        rs.arg_weight = w_arg
        rs.stance_weight = 0
        temp_precision = round(avg_topic_precision(rs, topics, k)[precision_pos], eta)
        if temp_precision > best_precision:
            best_weights = (w_top, w_arg, 1 - w_top - w_arg)
            best_precision = temp_precision

        took = datetime.datetime.now() - then
        left = took * i
        log.info('%s|%s w:%s took:%s, ~%s left', round(best_precision, eta),
                 round(temp_precision, eta), best_weights, took, left)

    log.info('Found best weights %s with precision %s', best_weights, best_precision)
    return best_weights


def main():
    model_name = 'test_1'
    rs = get_retrieval_system(Configuration(), model_name + "_argument", model_name + "_stance")
    topics = [Topic.get(t) for t in [2, 4, 8, 21, 27, 33, 36, 37, 40, 43, 45, 48]]
    find_rs_weights(rs, topics)
