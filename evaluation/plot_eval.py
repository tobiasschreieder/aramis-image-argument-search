import datetime
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from indexing import Topic
from retrieval import ArgumentModel, StanceModel, NNStanceModel

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
    elif len(topics) <= 20:
        rows = 4
        cols = 5
    else:
        rows = 4
        cols = 5

        for el in [36, 45, 37, 43]:
            try:
                topics.remove(el)
            except ValueError:
                pass
        if len(topics) > 20:
            topics = topics[:20]
        print('Cant plot more than 20 topics in one plot, tried %s', len(topics))

    k = 50
    round_int = 4

    sub_titel = []
    for t in topics:
        topic_title = Topic.get(t).title
        topic_title = (topic_title[:27] + '..') if len(topic_title) > 30 else topic_title
        sub_titel.append(f'Topic {Topic.get(t).number} - {topic_title}')
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, x_title='Score', y_title='Count',
                        vertical_spacing=0.05, horizontal_spacing=0.01, subplot_titles=sub_titel)
    a = np.array(topics)
    a.resize((rows, cols), refcheck=False)

    avg_precision = [0, 0]

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

            df = calc_topic_scores(model, topic, infos[0])

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            for i in [2, 3, 4]:
                x = df.loc[(df['value'] == infos[i][0]), 'score']
                fig.add_histogram(x=x, name=infos[i][0], showlegend=show_legend, row=row + 1, col=col + 1,
                                  legendgroup=infos[i][0], marker={'color': infos[i][1]})

            if infos[0] == 'argument':
                df_20 = df.nlargest(k, 'score', keep='first')
                precision_20 = df_20.loc[df_20['value'] == 'STRONG', 'value'].count() / k
                precision_20_both = df_20.loc[(df_20['value'] == 'STRONG') | (df_20['value'] == 'WEAK'),
                                              'value'].count() / k
                precision_text = f'PStrong@{k}: {round(precision_20, round_int)}<br>' \
                                 f'Relevant: {df.loc[df["value"] == "STRONG", "value"].count()}'
                count_both = df.loc[(df["value"] == "STRONG") | (df["value"] == "WEAK"), "value"].count()
                precision_text_both = f'PBoth@{k}: {round(precision_20_both, round_int)}<br>' \
                                      f'Relevant: {count_both}'
                avg_precision[0] += precision_20
                avg_precision[1] += precision_20_both
                fig.add_annotation(row=row + 1, col=col + 1, text=precision_text,
                                   xref='x domain', yref='y domain', showarrow=False, x=0.95, y=0.95)
                fig.add_annotation(row=row + 1, col=col + 1, text=precision_text_both,
                                   xref='x domain', yref='y domain', showarrow=False, x=0.05, y=0.95)
            elif infos[0] == 'stance':
                pro_scores = df.nlargest(k, 'score', keep='first')
                con_scores = df.nsmallest(k, 'score', keep='first')
                p_pro = pro_scores.loc[pro_scores['value'] == 'PRO', 'value'].count() / k
                p_con = con_scores.loc[con_scores['value'] == 'CON', 'value'].count() / k
                precision_text = f'PStrong@{k}:<br>Pro: {round(p_pro, round_int)}<br>Con: {round(p_con, round_int)}' \
                                 f'<br>Avg: {round((p_pro+p_con)/2, round_int)}<br>'
                avg_precision[0] += p_pro
                avg_precision[1] += p_con
                rel_pro = df.loc[df["value"] == "PRO", "value"].count()
                rel_con = df.loc[df["value"] == "CON", "value"].count()
                fig.add_annotation(row=row + 1, col=col + 1, text=precision_text,
                                   xref='x domain', yref='y domain', showarrow=False, x=0.95, y=0.95)
                fig.add_annotation(row=row + 1, col=col + 1, text=f'Relevant:<br>Pro: {rel_pro}<br>Con: {rel_con}',
                                   xref='x domain', yref='y domain', showarrow=False, x=0.05, y=0.95)

    if infos[0] == 'argument':
        precision_title = f'<br><sup>Strong@{k}: {round((avg_precision[0])/len(topics), round_int)}, ' \
                          f'Both@{k}: {round((avg_precision[1])/len(topics), round_int)}</sup>'
    else:
        pro = (avg_precision[0]+1)/len(topics)
        con = (avg_precision[1]+1)/len(topics)
        avg = (pro+con) / 2
        precision_title = f'<br><sup>Precision@{k}: Pro {round(pro, round_int)}, Con {round(con, round_int)}, ' \
                          f'Avg {round(avg, round_int)}</sup>'

    fig.update_layout(title=f'{infos[1]} Scoring {precision_title}')

    return fig


def plot_stance_confusion(model, topics: List[int]) -> go.Figure:
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
    elif len(topics) <= 20:
        rows = 4
        cols = 5
    else:
        rows = 4
        cols = 5

        for el in [36, 45, 37, 43]:
            try:
                topics.remove(el)
            except ValueError:
                pass
        if len(topics) > 20:
            topics = topics[:20]
        print('Cant plot more than 20 topics in one plot, tried %s', len(topics))

    infos = ('stance', 'Stance', ('PRO', plotly_color[2]), ('NEUTRAL', plotly_color[0]),
             ('CON', plotly_color[1]), 'color_mood, image_text_sentiment, html_sentiment')

    sub_titel = []
    for t in topics:
        topic_title = Topic.get(t).title
        topic_title = (topic_title[:30] + '..') if len(topic_title) > 32 else topic_title
        sub_titel.append(f'Topic {Topic.get(t).number} - {topic_title}')
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True, shared_xaxes=True, x_title='True Eval',
                        y_title='Predicted Eval',
                        vertical_spacing=0.06, horizontal_spacing=0.02, subplot_titles=sub_titel)
    a = np.array(topics)
    a.resize((rows, cols), refcheck=False)

    avg_error = 0
    avg_count = 0

    for row in range(rows):
        for col in range(cols):
            t = a[row, col]

            show_legend = False
            if row == 0 and col == 0:
                show_legend = True

            if t == 0:
                fig.add_histogram(x=[], name='', showlegend=show_legend,
                                  row=row + 1, col=col + 1, legendgroup='')
                continue
            topic = Topic.get(t)

            df = calc_topic_scores(model, topic, infos[0])

            # show_legend = False
            # if row == 0 and col == 0:
            #     show_legend = True

            x = ['CON', 'NEUTRAL', 'PRO']
            y = [-1, 0, 1]
            z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for j, eval_val in enumerate(x):
                count = df.loc[(df['value'] == eval_val), 'value'].count()
                for i, score in enumerate(y):
                    z[i][j] = df.loc[(df['value'] == eval_val) & (df['score'] == score), 'value'].count() / count
                    if i != j:
                        avg_count += 1
                        avg_error += z[i][j]
            z.reverse()
            y = x.copy()
            y.reverse()
            fig.add_heatmap(z=z, x=x, y=y, texttemplate='%{z:.4f}', colorscale='Blues', showscale=False,
                            row=row + 1, col=col + 1,)

    avg_error /= avg_count
    avg_error *= 2
    fig.update_layout(title=f'{infos[1]} Scoring Confusion matrix<br><sup>Avg Error: {round(avg_error, 4)}</sup>')

    return fig
