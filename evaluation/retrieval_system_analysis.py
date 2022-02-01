import datetime
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic, SpacyPreprocessor
from retrieval import RetrievalSystem, TopicRankingDirichlet, NNArgumentModel, NNStanceModel, StandardStanceModel, StandardArgumentModel
from .analysis_helper import get_relevant_eval, calc_precision_recall, get_topic_correct
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


def avg_topic_precision_recall(rs: RetrievalSystem, topics: List[Topic], k, strong: bool = True,
                               filter_topic: bool = True) -> Tuple[float, float, float, float, float, float]:
    pro_error_p = 0
    con_error_p = 0
    pro_error_r = 0
    con_error_r = 0
    for topic in topics:
        relevant_p, relevant_c = get_relevant_eval(topic, strong)
        if filter_topic:
            relevant_topic = get_topic_correct(topic)
            result_p, result_c = rs.query(topic.title, topic=topic)
            result_p = [s[0] for s in result_p if s[0] in relevant_topic][:k]
            result_c = [s[0] for s in result_c if s[0] in relevant_topic][:k]
        else:
            result_p, result_c = rs.query(topic.title, top_k=k, topic=topic)
            result_p = [s[0] for s in result_p]
            result_c = [s[0] for s in result_c]
        pp, pr = calc_precision_recall(result_p, relevant_p)
        cp, cr = calc_precision_recall(result_c, relevant_c)
        pro_error_p += pp
        con_error_p += cp
        pro_error_r += pr
        con_error_r += cr
    pro_error_p /= len(topics)
    con_error_p /= len(topics)
    pro_error_r /= len(topics)
    con_error_r /= len(topics)
    return pro_error_p, con_error_p, (pro_error_p + con_error_p) / 2, \
        pro_error_r, con_error_r, (pro_error_r + con_error_r) / 2


def avg_baseline_precision_recall(tm: TopicRankingDirichlet, topics: List[Topic], k, strong: bool = True,
                                  filter_topic: bool = True) -> Tuple[float, float, float, float, float, float]:
    pro_error_p = 0
    con_error_p = 0
    pro_error_r = 0
    con_error_r = 0
    prep = SpacyPreprocessor()
    for topic in topics:
        relevant_p, relevant_c = get_relevant_eval(topic, strong)
        pro_query = prep.preprocess(topic.title + ' good')
        con_query = prep.preprocess(topic.title + ' anti')
        if filter_topic:
            relevant_topic = get_topic_correct(topic)

            result_p: pd.DataFrame = tm.query(pro_query)
            result_c: pd.DataFrame = tm.query(con_query)
            result_p_list = [i for i in result_p.index if i in relevant_topic][:k]
            result_c_list = [i for i in result_c.index if i in relevant_topic][:k]
        else:
            result_p = tm.query(pro_query, top_k=k)
            result_c = tm.query(con_query, top_k=k)
            result_p_list = [s[0] for s in result_p]
            result_c_list = [s[0] for s in result_c]
        pp, pr = calc_precision_recall(result_p_list, relevant_p)
        cp, cr = calc_precision_recall(result_c_list, relevant_c)
        pro_error_p += pp
        con_error_p += cp
        pro_error_r += pr
        con_error_r += cr
    pro_error_p /= len(topics)
    con_error_p /= len(topics)
    pro_error_r /= len(topics)
    con_error_r /= len(topics)
    return pro_error_p, con_error_p, (pro_error_p + con_error_p) / 2, \
        pro_error_r, con_error_r, (pro_error_r + con_error_r) / 2


def binary_search_rs_weight(rs: RetrievalSystem, topics: List[Topic], strong: bool = True, k: int = 20,
                            eta: int = 5) -> None:
    logging.getLogger().setLevel(logging.INFO)

    change = True

    cur_range = [0, 0.5, 1]
    cur_precision = [0, 0, 0]

    for j, w in enumerate(cur_range):
        rs.topic_weight = w
        rs.arg_weight = 1 - w
        cur_precision[j] = round(avg_topic_precision_recall(rs, topics, k, strong=strong)[2], eta)

    i = 6
    while change:
        log.info('%s Cur Range: %s, Cur Precision: %s', i, cur_range, cur_precision)
        then = datetime.datetime.now()
        new_range = [0, 0]
        new_precision = [0, 0]

        new_range[0] = (cur_range[1] - cur_range[0]) / 2 + cur_range[0]
        new_range[1] = (cur_range[2] - cur_range[1]) / 2 + cur_range[1]
        log.info('New Range: %s', new_range)
        for j in [0, 1]:
            rs.topic_weight = new_range[j]
            rs.arg_weight = 1 - new_range[j]
            new_precision[j] = round(avg_topic_precision_recall(rs, topics, k, strong=strong)[2], eta)

        log.info('New precision: %s', new_precision)

        if new_precision[0] > new_precision[1]:
            cur_range[2] = cur_range[1]
            cur_range[1] = new_range[0]
            cur_precision[2] = cur_precision[1]
            cur_precision[1] = new_precision[0]
        else:
            cur_range[0] = cur_range[1]
            cur_range[1] = new_range[1]
            cur_precision[0] = cur_precision[1]
            cur_precision[1] = new_precision[1]

        i -= 1
        took = datetime.datetime.now() - then
        left = took * i
        log.info('took:%s, ~%s left', took, left)

        if i < 0:
            break
    log.info('End Cur Range: %s, Cur Precision: %s', cur_range, cur_precision)


def find_rs_weights(rs: RetrievalSystem, topics: List[Topic], strong: bool = True, optimum: str = 'both', k: int = 20,
                    eta: int = 5) -> Tuple[float, float, float]:
    if optimum.lower() == 'pro':
        precision_pos = 0
    elif optimum.lower() == 'con':
        precision_pos = 1
    else:
        precision_pos = 2

    # best_precision = 0.1895
    # best_weights = (0.33, 0.66, 1 - 0.33 - 0.66)
    best_precision = 0
    best_weights = (0, 1)
    i = 5
    for w_top in np.linspace(0, 1, i):
        w_arg = 1 - w_top
        i -= 1
        log.info(f'cur w:%s', (w_top, w_arg))
        then = datetime.datetime.now()
        rs.topic_weight = w_top
        rs.arg_weight = w_arg
        temp_precision = round(avg_topic_precision_recall(rs, topics, k, strong=strong)[precision_pos], eta)
        if temp_precision > best_precision:
            best_weights = (w_top, w_arg)
            best_precision = temp_precision

        took = datetime.datetime.now() - then
        left = took * i
        log.info('%s|%s w:%s took:%s, ~%s left', round(best_precision, eta),
                 round(temp_precision, eta), best_weights, took, left)

    log.info('Found best weights %s with precision %s', best_weights, best_precision)
    return best_weights


def eval_standard_model():
    load_indices()
    rs = RetrievalSystem(
        tq_index.prep,
        topic_model=TopicRankingDirichlet(
            t_indexes=topic_indexes, tq_index=tq_index, alpha=1000, tq_alpha=1000
        ),
        argument_model=StandardArgumentModel(findex),
        stance_model=StandardStanceModel(findex),
        topic_weight=0.454,
    )
    log.info('StandardModel')
    # topics = [Topic.get(t) for t in [1, 2, 4, 8, 10, 20, 21, 22, 40, 47]]
    # binary_search_rs_weight(rs, topics)
    print_precision_recall(rs)


def eval_nn_model():
    load_indices()
    rs = RetrievalSystem(
        tq_index.prep,
        topic_model=TopicRankingDirichlet(
            t_indexes=topic_indexes, tq_index=tq_index, alpha=1000, tq_alpha=1000
        ),
        argument_model=NNArgumentModel(findex, 'test_final_2'),
        stance_model=NNStanceModel(findex, 'test_final_2'),
        topic_weight=1,
    )
    log.info('NNModel')
    topics = [Topic.get(t) for t in [1, 2, 4, 8, 10, 20, 21, 22, 40, 47]]
    binary_search_rs_weight(rs, topics)
    print_precision_recall(rs)


def eval_baseline():
    load_indices()
    topic_model = TopicRankingDirichlet(
        t_indexes=topic_indexes, tq_index=tq_index, alpha=1000, tq_alpha=1000
    )
    topics = [Topic.get(t) for t in [9, 27, 31, 33]]
    strong = avg_baseline_precision_recall(topic_model, topics, 20, strong=True, filter_topic=True)
    both = avg_baseline_precision_recall(topic_model, topics, 20, strong=False, filter_topic=True)
    log.info('PStrongPRO@20: %s', round(strong[0], 5))
    log.info('PStrongCON@20: %s', round(strong[1], 5))
    log.info('PStrongAVG@20: %s', round(strong[2], 5))
    log.info('RStrongPRO@20: %s', round(strong[3], 5))
    log.info('RStrongCON@20: %s', round(strong[4], 5))
    log.info('RStrongAVG@20: %s', round(strong[5], 5))
    log.info('PBothPRO@20: %s', round(both[0], 5))
    log.info('PBothCON@20: %s', round(both[1], 5))
    log.info('PBothAVG@20: %s', round(both[2], 5))
    log.info('RBothPRO@20: %s', round(both[3], 5))
    log.info('RBothCON@20: %s', round(both[4], 5))
    log.info('RBothAVG@20: %s', round(both[5], 5))


def print_precision_recall(rs: RetrievalSystem):
    topics = [Topic.get(t) for t in [9, 27, 31, 33]]
    strong = avg_topic_precision_recall(rs, topics, 20, strong=True, filter_topic=True)
    both = avg_topic_precision_recall(rs, topics, 20, strong=False, filter_topic=True)
    log.info('PStrongPRO@20: %s', round(strong[0], 5))
    log.info('PStrongCON@20: %s', round(strong[1], 5))
    log.info('PStrongAVG@20: %s', round(strong[2], 5))
    log.info('RStrongPRO@20: %s', round(strong[3], 5))
    log.info('RStrongCON@20: %s', round(strong[4], 5))
    log.info('RStrongAVG@20: %s', round(strong[5], 5))
    log.info('PBothPRO@20: %s', round(both[0], 5))
    log.info('PBothCON@20: %s', round(both[1], 5))
    log.info('PBothAVG@20: %s', round(both[2], 5))
    log.info('RBothPRO@20: %s', round(both[3], 5))
    log.info('RBothCON@20: %s', round(both[4], 5))
    log.info('RBothAVG@20: %s', round(both[5], 5))


def main():
    eval_standard_model()

    # model_name = 'test_3'
    # rs = get_retrieval_system(Configuration(), model_name, model_name)
    # topics = [Topic.get(t) for t in [2, 4, 8, 21, 27, 33, 36, 37, 40, 43, 45, 48]]
    #
    # find_rs_weights(rs, topics)
