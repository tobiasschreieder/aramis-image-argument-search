from typing import Tuple, List, Dict
from sklearn.model_selection import RepeatedKFold
import numpy as np

from evaluation import analysis_labeled_data


def get_valid_topics(topics: List, model: str) -> Tuple[List, List]:
    """
    Get Topics with enough argumentative or Stance "pro" and "con" images
    :param topics: List with all labeled topics
    :param model: "Argument" or "Stance"
    :return: valid Topics for chosen model as List and skipped topics as List
    """
    topics_skip = []
    topics_valid = []

    analysis, eda = analysis_labeled_data.exploratory_data_analysis()

    if model == "Argument":
        for i in topics:
            if analysis[i]["count_argumentative_strong"] >= 20:
                topics_valid.append(i)
            else:
                topics_skip.append(i)

    if model == "Stance":
        for i in topics:
            if analysis[i]["count_stance_pro"] >= 20 and analysis[i]["count_stance_con"] >= 20:
                topics_valid.append(i)
            else:
                topics_skip.append(i)

    return topics_valid, topics_skip


def repeated_kfold(valid_topics: List) -> Dict:
    """
    Get Train- and Testsplits of valid Topics with Repeated KFold
    :param valid_topics: List with valid topics
    :return: all folds as Dict
    """
    folds = dict()

    rkf = RepeatedKFold(n_splits=5, n_repeats=2)

    counter = 1
    for train, test in rkf.split(valid_topics):
        train_topics = list(np.take(valid_topics, train))
        test_topics = list(np.take(valid_topics, test))

        split = dict()
        split.setdefault("train", train_topics)
        split.setdefault("test", test_topics)

        folds.setdefault(counter, split)

        counter += 1

    return folds


def write_folds(folds: Dict):
    """
    Write folds as TXT-File
    :param folds: folds as Dict
    """
    text = []

    for fold in folds:
        text.append("Run " + str(fold))
        text.append("train_topics = " + str(folds[fold]["train"]))
        text.append("test_topics = " + str(folds[fold]["test"]))
        text.append("\n")

    with open("folds.txt", 'w') as output:
        for row in text:
            output.write(str(row) + '\n')
