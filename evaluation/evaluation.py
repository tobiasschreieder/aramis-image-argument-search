import logging
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np

from config import Config
from indexing import Topic, DataEntry, FeatureIndex

cfg = Config.get()
log = logging.getLogger('Evaluation')


class Argumentative(Enum):
    NONE = 0
    WEAK = 1
    STRONG = 2


class Stance(Enum):
    PRO = 0
    NEUTRAL = 1
    CON = 2


eval_file = cfg.data_location.joinpath(Path('image_eval.txt'))
if eval_file.exists():
    df = pd.read_csv(eval_file, sep=' ')
else:
    df = pd.DataFrame(columns=['image_id', 'user', 'Topic', 'Topic_correct', 'Argumentative', 'Stance'])

df.astype(dtype={
            'image_id': pd.StringDtype(),
            'user': pd.StringDtype(),
            'Topic': np.int,
            'Topic_correct': np.bool,
            'Argumentative': pd.StringDtype(),
            'Stance': pd.StringDtype(),
        })
df.set_index(['image_id', 'user', 'Topic'], inplace=True)


def get_df() -> pd.DataFrame:
    return df.copy()


def save_df():
    df.to_csv(eval_file, sep=' ')


def clean_image_eval(data):
    """
    Clean wrong data in image_eval.txt
    Set "Argumentative" to "NONE" and "Stance" to "NEUTRAL" if topic is not relevant
    :param data: Dataframe of image_eval
    :return: cleaned data as Dataframe
    """
    data = data.reset_index()

    wrong_argument = 0
    wrong_stance = 0

    for i in data.index:
        column = data.loc[i]

        topic = column.loc["Topic_correct"]
        argument = column.loc["Argumentative"]
        stance = column.loc["Stance"]

        if not topic:
            if argument != "NONE":
                wrong_argument += 1
                data.at[i, "Argumentative"] = "NONE"

            if stance != "NEUTRAL":
                wrong_stance += 1
                data.at[i, "Stance"] = "NEUTRAL"

    print("Cleaned Argumentative values:", str(wrong_argument))
    print("Cleaned Stance values:", str(wrong_stance))

    data = data.set_index(['image_id', 'user', 'Topic'])

    return data


def has_eval(image_id: str, topic: int = None) -> bool:
    if topic:
        try:
            return len(df.loc[(image_id, slice(None), topic), :]) > 0
        except KeyError:
            return False
    return image_id in df.index.get_level_values(0)


def get_image_to_eval(topic: Topic) -> DataEntry or None:
    for image in topic.get_image_ids():
        if has_eval(image, topic.number):
            continue
        return DataEntry.load(image)
    return None


def get_eval(image_id: str, topic: int) -> Tuple[int, Argumentative, Stance] or None:
    if has_eval(image_id):
        temp = df.loc[(image_id, slice(None), topic), :]
        return (temp.loc[temp.index[0], 'Topic'],
                Argumentative[temp.loc[temp.index[0], 'Argumentative']],
                Stance[temp.loc[temp.index[0], 'Stance']])
    return None


def get_evaluations(image_id: str, topic: int) -> Dict[str, Tuple[int, Argumentative, Stance]] or None:
    if has_eval(image_id, topic):
        temp = df.loc[(image_id, slice(None), topic), :]
        evals = []
        for user in temp.index:
            evals.append((temp.loc[user, 'Topic'],
                          Argumentative[temp.loc[user, 'Argumentative']],
                          Stance[temp.loc[user, 'Stance']]))
        return evals
    return None


def save_eval(image_id: str, user: str, topic: int, topic_correct: bool, arg: Argumentative, stance: Stance) -> None:
    df.loc[(image_id, user, topic), :] = [topic_correct, arg.name, stance.name]
    save_df()
    log.debug('Saved evaluation for %s %s %s: %s %s %s', image_id, user, topic, topic_correct, arg, stance)


def get_model_data_arg(topics: List[Topic], fidx: FeatureIndex) -> pd.DataFrame:
    data = fidx.dataframe.copy()
    data['arg_eval'] = 0
    data['topic'] = 0
    for topic in topics:
        t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]
        data.loc[t_df.loc[t_df['Topic_correct'], :].index.unique(0), 'topic'] = topic.number
        data.loc[t_df.loc[
                 (t_df['Topic_correct'] & (t_df['Argumentative'] == 'STRONG')), :].index.unique(0), 'arg_eval'] = 1
        data.loc[t_df.loc[
                 (t_df['Topic_correct'] & (t_df['Argumentative'] == 'WEAK')), :].index.unique(0), 'arg_eval'] = 0.5
    return data.loc[data['topic'] > 0, :]


def get_model_data_stance(topics: List[Topic], fidx: FeatureIndex) -> pd.DataFrame:
    data = fidx.dataframe.copy()
    data['topic'] = 0
    for topic in topics:
        t_df: pd.DataFrame = get_df().loc[(slice(None), slice(None), topic.number), :]
        data.loc[t_df.loc[t_df['Topic_correct'], :].index.unique(0), 'topic'] = topic.number
        data.loc[t_df.loc[
                 (t_df['Topic_correct'] & (t_df['Stance'] == 'PRO')), :].index.unique(0), 'stance_eval'] = (1, 0)
        data.loc[t_df.loc[
                 (t_df['Topic_correct'] & (t_df['Stance'] == 'NEUTRAL')), :].index.unique(0), 'stance_eval'] = (0, 0)
        data.loc[t_df.loc[
                 (t_df['Topic_correct'] & (t_df['Stance'] == 'CON')), :].index.unique(0), 'stance_eval'] = (0, 1)
    return data.loc[(data['topic'] > 0), :]
