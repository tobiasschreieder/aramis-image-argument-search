import datetime
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from indexing import Topic
from retrieval import ArgumentModel, StanceModel

from .analysis_helper import calc_topic_scores

log = logging.getLogger('plot_eval')


plotly_color = qualitative.Plotly


def plot_arg_scoring_eval(model: ArgumentModel, topics: List[int]) -> go.Figure:
    infos = ('argument', 'Argumentative', ('NONE', plotly_color[5]), ('WEAK', plotly_color[6]),
             ('STRONG', plotly_color[7]), 'diagramm, text_sentiment, text, html_sentiment')
    fig = plot_scoring_eval(model, topics, infos)
    path = Path('plots')
    path.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig.write_image(path.joinpath(f'arg_scoring_{now}.png'), width=1600, height=900)
    return fig


def plot_stance_scoring_eval(model: StanceModel, topics: List[int]) -> go.Figure:
    infos = ('stance', 'Stance', ('PRO', plotly_color[2]), ('NEUTRAL', plotly_color[0]),
             ('CON', plotly_color[1]), 'color_mood, image_text_sentiment, html_sentiment')
    fig = plot_scoring_eval(model, topics, infos)
    path = Path('plots')
    path.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig.write_image(path.joinpath(f'stance_scoring_{now}.png'), width=1600, height=900)
    return fig


def plot_scoring_eval(model, topics: List[int],
                      infos: Tuple[str, str, Tuple[str, str], Tuple[str, str], Tuple[str, str], str]) -> go.Figure:
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

            df = calc_topic_scores(model, topic, infos[0])

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            for i in [2, 3, 4]:
                x = df.loc[(df['value'] == infos[i][0]), 'score']
                fig.add_histogram(x=x, name=infos[i][0], showlegend=show_legend, row=row + 1, col=col + 1,
                                  legendgroup=infos[i][0], marker={'color': infos[i][1]})

    fig.update_layout(title=f'{infos[1]} Scoring')

    return fig
