import datetime
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from evaluation import get_df
from .configuration import Configuration
from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic, SpacyPreprocessor
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardArgumentModel, StandardStanceModel

log = logging.getLogger('analysis')


plotly_color = qualitative.Plotly
tq_index: TopicQueryTermIndex = None
topic_indexes: Dict[int, TopicTermIndex] = None
findex: FeatureIndex = None


def load_indices() -> None:
    global tq_index, topic_indexes, findex
    tq_index = TopicQueryTermIndex.load()
    topic_indexes = get_all_topic_indexes()
    findex = FeatureIndex.load(23158)


def get_retrieval_system(cfg: Configuration) -> RetrievalSystem:
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
    else:
        raise NotImplementedError('Cant plot more than 9 topics in one plot, tried %s', len(topics))

    sub_titel = [f'Topic {Topic.get(t).number} - {Topic.get(t).title}' for t in topics]
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, x_title='Score', y_title='Count',
                        vertical_spacing=0.05, horizontal_spacing=0.01, subplot_titles=sub_titel)
    a = np.array(topics)
    a.resize((rows, cols), refcheck=False)

    for row in range(rows):
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

            t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]

            t_df = t_df.where(t_df['Topic_correct']).dropna()
            query = SpacyPreprocessor().preprocess(topic.title)

            data = []
            score_df = pd.DataFrame(index=t_df.index.get_level_values(0))
            model.query(query, score_df, weights=weights)

            for image in score_df.index:
                score = score_df.loc[image, infos[0]]
                temp = t_df.loc[(image, slice(None), topic.number), [infos[1]]]
                if len(temp) != 1:
                    raise NotImplementedError('Found more than 1 evaluation to image %s and Topic %s', image,
                                              topic.number)
                arg = temp.to_numpy()[0]
                data.append((score, arg))

            df = pd.DataFrame(data, columns=['score', 'value'])

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            for i in [2, 3, 4]:
                fig.add_histogram(x=df.loc[(df['value'] == infos[i][0]), 'score'], name=infos[i][0],
                                  showlegend=show_legend, row=row + 1, col=col + 1, legendgroup=infos[i][0],
                                  marker={'color': infos[i][1]})

    fig.update_layout(title=f'{infos[1]} Scoring with {weights} for {infos[5]}')

    return fig


def eval_all():
    # try all parameter in a defined range and save results
    # 3-fold (bootstrap) over topics (choose rdm 80% topics eval with rest) (3/1)
    pass

    # RetrievalSystem:
    #   - topic_weight
    #   - argument_weight
    #   - prefetch_top_k
    #
    # TopicRankingDirichlet
    #   - topic query alpha
    #   - alpha
    #
    # StandardArgumentModel
    #   - usage of features
    #
    # StandardStanceModel
    #   - usage of features

    # Needed:
    #   metrik retrieval result -> score
    #       -> precision/recall - f1-funktion (@k)
    #   arg/stance scoring
    #       -> verteilung ergebnisse ähnlich zu verteilung evaluation

    # Relevant = Topic: True, Arg: Strong, Stance: Pro/Con

    # function configuration in f1 out

    # 1. f1 system eval = avg over all topics (5,5)
    # calc eval for each parameter range, save best arg
    # best cfg = all best parameters

    # 2. eval single models using eval_scoring()
    # find best parameter for single model
    # do rs eval


def main():
    # load_indices()
    # rs = get_retrieval_system(Configuration())
    # log.debug('%s', topic_precision_recall(rs, Topic.get(33), 50))
    # plot_topic_precision_recall_at_k(Configuration(), [33], max_k=20)

    findex = FeatureIndex.load(23158)

    plot_arg_scoring_eval(StandardArgumentModel(findex), [21, 27, 33, 48], (1, 1, 1, 1)).show()
    plot_stance_scoring_eval(StandardStanceModel(findex), [21, 27, 33, 48], (1, 1, 1)).show()
