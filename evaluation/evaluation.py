import logging
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np

from config import Config
from indexing import Topic, DataEntry

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


def save_df():
    df.to_csv(eval_file, sep=' ')


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