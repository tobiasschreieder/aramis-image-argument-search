import logging
from typing import Tuple, List

import pandas as pd

from evaluation import get_df
from indexing import Topic, SpacyPreprocessor
from retrieval import StanceModel, ArgumentModel

log = logging.getLogger('analysis_helper')


def calc_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall)


def calc_precision_recall(results: List[str], relevant: List[str]) -> Tuple[float, float]:
    if len(results) == 0:
        return 0, 0
    res_set = set(results)
    rel_set = set(relevant)
    a = len(res_set.intersection(rel_set))
    precision = a / len(res_set)
    recall = a / len(rel_set)
    return precision, recall


def get_relevant_eval(topic: Topic) -> Tuple[List[str], List[str]]:
    # Get relevant images pro
    t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]
    t_df['relevant_pro'] = 1
    t_df['relevant_pro'].where(
        (t_df['Topic_correct'] & (t_df['Argumentative'] == 'STRONG') & (t_df['Stance'] == 'PRO')),
        other=0, inplace=True,
    )
    relevant_p = t_df.loc[t_df['relevant_pro'] == 1].index.get_level_values(0).to_list()

    # Get relevant images con
    t_df['relevant_con'] = 1
    t_df['relevant_con'].where(
        (t_df['Topic_correct'] & (t_df['Argumentative'] == 'STRONG') & (t_df['Stance'] == 'CON')),
        other=0, inplace=True,
    )
    relevant_c = t_df.loc[t_df['relevant_con'] == 1].index.get_level_values(0).to_list()
    return relevant_p, relevant_c


def calc_topic_scores(model, topic: Topic, score_type: str) -> pd.DataFrame:
    if score_type == 'arg' or score_type == 'argument':
        score_name = 'argument'
        eval_name = 'Argumentative'
    else:
        score_name = 'stance'
        eval_name = 'Stance'
    t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]

    t_df = t_df.where(t_df['Topic_correct']).dropna()
    query = SpacyPreprocessor().preprocess(topic.title)

    data = []
    score_df = pd.DataFrame(index=t_df.index.get_level_values(0).unique())
    model.query(query, score_df)

    for image in score_df.index:
        score = score_df.loc[image, score_name]
        temp = t_df.loc[(image, slice(None), topic.number), eval_name]
        if len(temp) != 1:
            raise NotImplementedError('Found more than 1 evaluation to image %s and Topic %s', image,
                                      topic.number)
        arg = temp.to_numpy()[0]
        data.append((score, arg))

    return pd.DataFrame(data, columns=['score', 'value'])


def avg_precision_arg_error(model: ArgumentModel, topics: List[Topic], k: int = 20) -> float:
    avg_error = 0
    for topic in topics:
        scores = calc_topic_scores(model, topic, 'arg').nlargest(k, 'score', keep='all')
        relevant_result = scores.loc[scores['value'] == 'STRONG', 'value'].count()
        avg_error += relevant_result / k
    return avg_error / len(topics)


def avg_precision_stance_error(model: StanceModel, topics: List[Topic], k: int = 20) -> Tuple[float, float, float]:
    pro_error = 0
    con_error = 0
    for topic in topics:
        scores = calc_topic_scores(model, topic, 'stance')
        pro_scores = scores.nlargest(k, 'score', keep='first')
        con_scores = scores.nsmallest(k, 'score', keep='first')
        relevant_pro = pro_scores.loc[pro_scores['value'] == 'PRO', 'value'].count()
        relevant_con = con_scores.loc[con_scores['value'] == 'CON', 'value'].count()
        pro_error += relevant_pro / k
        con_error += relevant_con / k
    pro_error /= len(topics)
    con_error /= len(topics)
    return pro_error, con_error, (pro_error + con_error) / 2
