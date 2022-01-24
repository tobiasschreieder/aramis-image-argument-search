import datetime
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Callable, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from evaluation import get_df
from .configuration import Configuration
from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic, SpacyPreprocessor
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardArgumentModel, \
    StandardStanceModel, NNArgumentModel

log = logging.getLogger('analysis')


plotly_color = qualitative.Plotly
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


def get_retrieval_system(cfg: Configuration) -> RetrievalSystem:
    load_indices()
    return RetrievalSystem(
        tq_index.prep,
        topic_model=TopicRankingDirichlet(
            t_indexes=topic_indexes, tq_index=tq_index, alpha=cfg.alpha, tq_alpha=cfg.tq_alpha
        ),
        argument_model=StandardArgumentModel(findex),
        stance_model=StandardStanceModel(findex),
        topic_weight=cfg.topic_weight,
        argument_weight=cfg.argument_weight,
        prefetch_top_k=cfg.prefetch_top_k,
    )


def calc_precision_recall(results: List[str], relevant: List[str]) -> Tuple[float, float]:
    res_set = set(results)
    rel_set = set(relevant)
    log.debug('equal len res: %s', len(res_set) == len(results))
    log.debug('equal len rel: %s', len(rel_set) == len(relevant))
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


def topic_precision_recall(rs: RetrievalSystem, topic: Topic, k: int) -> Tuple[datetime.timedelta,
                                                                               Tuple[float, float],
                                                                               Tuple[float, float]]:
    then = datetime.datetime.now()
    result_p, result_c = rs.query(topic.title, top_k=k)
    now = datetime.datetime.now()

    result_p = [s[0] for s in result_p]
    result_c = [s[0] for s in result_c]

    relevant_p, relevant_c = get_relevant_eval(topic)

    return now-then, calc_precision_recall(result_p, relevant_p), calc_precision_recall(result_c, relevant_c)


def plot_topic_precision_recall_at_k(cfg: Configuration, topics: List[int], max_k: int = 50) -> None:
    # Configuration mit beim Bild
    # speicher cfg in file with name, print name on pic
    # avg over all given topics or all topcis in one picture
    topics = [Topic.get(topic) for topic in topics]
    df = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=['topic', 'k']),
                      columns=['pro_precision', 'pro_recall', 'con_precision', 'con_recall', 'precision', 'recall'])
    rs = get_retrieval_system(cfg)
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


def plot_arg_scoring_eval(model: StandardArgumentModel, topics: List[int],
                          weights: Tuple[float, float, float, float] = None) -> go.Figure:
    infos = ('argument', 'Argumentative', ('NONE', plotly_color[5]), ('WEAK', plotly_color[6]),
             ('STRONG', plotly_color[7]), 'diagramm, text_sentiment, text, html_sentiment')
    fig = plot_scoring_eval(model, topics, infos, weights)
    path = Path('plots')
    path.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig.write_image(path.joinpath(f'arg_scoring_{now}.png'), width=1600, height=900)
    return fig


def plot_stance_scoring_eval(model: StandardStanceModel, topics: List[int],
                             weights: Tuple[float, float, float] = None) -> go.Figure:
    infos = ('stance', 'Stance', ('PRO', plotly_color[2]), ('NEUTRAL', plotly_color[0]),
             ('CON', plotly_color[1]), 'color_mood, image_text_sentiment, html_sentiment')
    fig = plot_scoring_eval(model, topics, infos, weights)
    path = Path('plots')
    path.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig.write_image(path.joinpath(f'stance_scoring_{now}.png'), width=1600, height=900)
    return fig


def plot_scoring_eval(model, topics: List[int],
                      infos: Tuple[str, str, Tuple[str, str], Tuple[str, str], Tuple[str, str], str],
                      weights: List[float]) -> go.Figure:
    if len(topics) <= 2:
        rows = 2
        cols = 1
    elif len(topics) <= 4:
        rows = 2
        cols = 2
    elif len(topics) <= 6:
        rows = 3
        cols = 2
    elif len(topics) <= 9:
        rows = 3
        cols = 3
    elif len(topics) <= 12:
        rows = 3
        cols = 4
    else:
        raise NotImplementedError('Cant plot more than 12 topics in one plot, tried %s', len(topics))

    sub_titel = [f'Topic {Topic.get(t).number} - {Topic.get(t).title}' for t in topics]
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, x_title='Score', y_title='Count',
                        vertical_spacing=0.05, horizontal_spacing=0.01, subplot_titles=sub_titel)
    a = np.array(topics)
    a.resize((rows, cols), refcheck=False)

    for row in range(rows):
        print("%s / %s" % (row, rows))
        for col in range(cols):
            t = a[row, col]

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            if t == 0:
                fig.add_histogram(x=[], name=infos[2][0], showlegend=show_legend,
                                  row=row + 1, col=col + 1, legendgroup=infos[2][0], marker={'color': infos[2][1]})
                continue
            topic = Topic.get(t)

            df = calc_topic_scores(model, weights, topic, infos[0], infos[1])

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            log.info('Topic %s error %s', topic.number, arg_error(df))

            for i in [2, 3, 4]:
                x = df.loc[(df['value'] == infos[i][0]), 'score']
                # log.info('Topic %s - %s: mean %s, std %s', topic.number, infos[i][0], x.mean(), x.std())
                fig.add_histogram(x=x, name=infos[i][0], showlegend=show_legend, row=row + 1, col=col + 1,
                                  legendgroup=infos[i][0], marker={'color': infos[i][1]})

    fig.update_layout(title=f'{infos[1]} Scoring with {weights} for {infos[5]}')

    return fig


def calc_topic_scores(model, weights: List[float], topic: Topic, score_name: str, eval_name: str) -> pd.DataFrame:
    t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]

    t_df = t_df.where(t_df['Topic_correct']).dropna()
    query = SpacyPreprocessor().preprocess(topic.title)

    data = []
    score_df = pd.DataFrame(index=t_df.index.get_level_values(0).unique())
    model.query(query, score_df, weights=weights)

    for image in score_df.index:
        score = score_df.loc[image, score_name]
        temp = t_df.loc[(image, slice(None), topic.number), eval_name]
        if len(temp) != 1:
            raise NotImplementedError('Found more than 1 evaluation to image %s and Topic %s', image,
                                      topic.number)
        arg = temp.to_numpy()[0]
        data.append((score, arg))

    return pd.DataFrame(data, columns=['score', 'value'])


def avg_arg_error(model, topics: List[Topic], weights: List[float]) -> float:
    avg_error = 0
    for topic in topics:
        avg_error += arg_error(calc_topic_scores(model, weights, topic, 'argument', 'Argumentative'))
    avg_error /= len(topics)
    return avg_error


def avg_precision_arg_error(model, topics: List[Topic], weights: List[float]) -> float:
    k = 20
    avg_error = 0
    for topic in topics:
        df = calc_topic_scores(model, weights, topic, 'argument', 'Argumentative')
        scores = df.nlargest(k, 'score', keep='all')
        relevant = scores.loc[scores['value'] == 'STRONG', 'value'].count()
        precision = relevant / k
        avg_error += 1 - precision
    avg_error /= len(topics)
    return avg_error


def avg_stance_error(model, topics: List[Topic], weights: List[float]) -> float:
    avg_err = 0
    for topic in topics:
        avg_err += stance_error(calc_topic_scores(model, weights, topic, 'stance', 'Stance'))
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

    # RetrievalSystem:
    #   - topic_weight
    #   - argument_weight
    #   - prefetch_top_k
    #
    # TopicRankingDirichlet
    #   - topic query alpha
    #   - alpha

    # Needed:
    #   metrik retrieval result -> score
    #       -> precision/recall - f1-funktion (@k)

    # function configuration in f1 out
    # 3-fold (bootstrap) over topics (choose rdm 80% topics eval with rest) (3/1)

    # 1. f1 system eval = avg over all topics (5,5)
    # calc eval for each parameter range, save best arg
    # best cfg = all best parameters

    # 2. eval single models using eval_scoring()
    # find best parameter for single model
    # do rs eval


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


def get_model_data_arg(topics: List[Topic], fidx: FeatureIndex) -> pd.DataFrame:
    df = fidx.dataframe.copy()
    df['arg_eval'] = 0
    df['topic'] = 0
    for topic in topics:
        t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]
        df.loc[t_df.loc[t_df['Topic_correct'], :].index.unique(0), 'topic'] = 1
        df.loc[t_df.loc[(t_df['Topic_correct'] & (t_df['Argumentative'] == 'STRONG')), :].index.unique(0), 'arg_eval'] = 1
        df.loc[t_df.loc[(t_df['Topic_correct'] & (t_df['Argumentative'] == 'WEAK')), :].index.unique(0), 'arg_eval'] = 0.5
    return df.loc[(df['topic'] == 1), :].drop('topic', axis=1)


def main():
    # rs = get_retrieval_system(Configuration())
    # log.debug('%s', topic_precision_recall(rs, Topic.get(4), 50))
    # plot_topic_precision_recall_at_k(Configuration(), [4], max_k=100)

    findex = FeatureIndex.load(23158)

    topics_no = [2, 4, 8, 21, 27, 33, 36, 40, 43, 45, 48]

    topics = [Topic.get(t) for t in topics_no]

    # arg_eval(findex, topics)
    # stance_eval(findex, topics)

    # t = get_model_data_arg(topics, findex)
    # print(t)

    # plot_arg_scoring_eval(StandardArgumentModel(findex), topics_no, [200001.0, 1010001.0, 1, 72221000001, -61419999999]).show()
    plot_arg_scoring_eval(NNArgumentModel(findex), topics_no, [200001.0, 1010001.0, 1, 72221000001, -61419999999]).show()
    # plot_stance_scoring_eval(StandardStanceModel(findex), topics_no, (1, 1, 1)).show()
