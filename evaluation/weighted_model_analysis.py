import logging
from typing import List, Callable, Any

import pandas as pd
from plotly.express.colors import qualitative

from indexing import FeatureIndex, Topic
from retrieval import StandardArgumentModel, StandardStanceModel
from .analysis_helper import calc_topic_scores

log = logging.getLogger('weight_analysis')


plotly_color = qualitative.Plotly


def avg_arg_error(model, topics: List[Topic]) -> float:
    avg_error = 0
    for topic in topics:
        avg_error += arg_error(calc_topic_scores(model, topic, 'arg'))
    avg_error /= len(topics)
    return avg_error


def avg_precision_arg_error(model, topics: List[Topic]) -> float:
    k = 20
    avg_error = 0
    for topic in topics:
        df = calc_topic_scores(model, topic, 'arg')
        scores = df.nlargest(k, 'score', keep='all')
        relevant = scores.loc[scores['value'] == 'STRONG', 'value'].count()
        precision = relevant / k
        avg_error += 1 - precision
    avg_error /= len(topics)
    return avg_error


def avg_stance_error(model, topics: List[Topic]) -> float:
    avg_err = 0
    for topic in topics:
        avg_err += stance_error(calc_topic_scores(model, topic, 'stance'))
    avg_err /= len(topics)
    return avg_err


def find_weight(model, error_func: Callable[[Any, List[Topic], List[float]], float], topics: List[Topic],
                weights: List[float], pos: int,
                decimal: float, max_iter: int = 10000, eta_dec: int = 7) -> float:
    best_error = round(error_func(model, topics, weights), eta_dec)
    best_weight = weights[pos]
    cur_add = True
    change = True
    i = 0
    while change:
        i += 1
        change = False
        if cur_add:
            weights[pos] += decimal
        else:
            weights[pos] -= decimal
        temp_error = round(error_func(model, topics, weights), eta_dec)
        if temp_error < best_error:
            best_weight = weights[pos]
            best_error = temp_error
            change = True
        else:
            weights[pos] = best_weight
            if cur_add:
                cur_add = False
                change = True

        log.debug('%s %s|%s w:%s add:%s, change:%s', i, round(best_error, eta_dec),
                  round(temp_error, eta_dec), weights[pos], cur_add, change)
        if i > max_iter:
            change = False

        if i > 10 and change:
            log.debug('Not found after 10 iteration, go one decimal up.')
            best_weight = find_weight(model, error_func, topics, weights, pos, decimal * 10, max_iter, eta_dec)
            best_error = round(error_func(model, topics, weights), eta_dec)
            cur_add = True
            change = True
            i = 0
            max_iter -= 10

    return best_weight


def find_weights(model, error_func: Callable[[Any, List[Topic], List[float]], float], topics: List[Topic],
                 start_dec: float, end_dec: float, start_weights: List[float], max_iter: int = 10000,
                 eta: int = 7) -> List[float]:
    dec = start_dec
    while dec > end_dec:
        for pos in range(len(start_weights)):
            log.debug('pos:%s, dec:%s, w:%s', pos, dec, start_weights)
            start_weights[pos] = find_weight(model, error_func, topics, start_weights, pos, dec, max_iter, eta_dec=eta)
        dec /= 10
    return start_weights


def arg_error(scores: pd.DataFrame):
    none_err = abs(scores.loc[scores['value'] == 'NONE', 'score'].mean() - 0)
    weak_err = abs(scores.loc[scores['value'] == 'WEAK', 'score'].mean() - 0.5)
    strong_err = abs(scores.loc[scores['value'] == 'STRONG', 'score'].mean() - 1)

    return 0.1 * none_err + 0.3 * weak_err + 0.6 * strong_err


def stance_error(scores: pd.DataFrame):
    con_err = abs(scores.loc[scores['value'] == 'CON', 'score'].mean() + 1)
    neutral_err = abs(scores.loc[scores['value'] == 'NEUTRAL', 'score'].mean() - 0)
    pro_err = abs(scores.loc[scores['value'] == 'PRO', 'score'].mean() - 1)

    return 0.4 * con_err + 0.2 * neutral_err + 0.4 * pro_err


def arg_eval(fidx: FeatureIndex, topics: List[Topic]):
    # weights = [-123999.0, 2002001.0, -999999, 1200001.0, -812999.0]
    # weights = [-117299.0, 2025201.0, -1007999.0, 1107001.0, -788999.0]
    # weights = [-117299.0, 2025201.0, -1007999.0, 1041901.0, -782299.0]
    # weights = [200001.0, 1010001.0, 1, 72221000001, -61419999999]
    weights = [1, 1, 1, 1, 1]
    find_weights(StandardArgumentModel(fidx), avg_precision_arg_error, topics,
                 start_dec=1000000, end_dec=1, start_weights=weights)


def stance_eval(fidx: FeatureIndex, topics: List[Topic]):
    weights = [1, 1, 1]
    find_weights(StandardStanceModel(fidx), avg_stance_error, topics,
                 start_dec=1000000, end_dec=1, start_weights=weights)


def main():
    findex = FeatureIndex.load(23158)

    topics_no = [2, 4, 8, 21, 27, 33, 36, 40, 43, 45, 48]

    topics = [Topic.get(t) for t in topics_no]

    arg_eval(findex, topics)
    stance_eval(findex, topics)

    # plot_arg_scoring_eval(StandardArgumentModel(findex), topics_no).show()
    # plot_stance_scoring_eval(StandardStanceModel(findex), topics_no).show()
