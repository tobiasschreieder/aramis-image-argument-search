import logging
from enum import Enum
from typing import Tuple, Dict

from config import Config
from indexing import Topic, DataEntry
from .eval_data import df, save_df

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
