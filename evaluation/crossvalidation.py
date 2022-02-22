from typing import Tuple, List, Dict
from sklearn.model_selection import RepeatedKFold
import numpy as np

from evaluation import analysis_labeled_data
from indexing import FeatureIndex, Topic, preprocessed_data, scale_data, NArgumentModel, NStanceModel
from .plot_eval import plot_arg_scoring_eval, plot_stance_confusion_eval
from retrieval import NNArgumentModel, NNStanceModel


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


def evaluation_nn_argument_image_split(runs: int) -> Dict:
    """
    Run Evaluation for NNArgument-Model over all topics with inter train- & test-split
    :param runs: number of runs
    :return: Dictionaries with results of runs
    """
    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]
    prep_data = preprocessed_data(findex, topics, train=True)
    data = scale_data(prep_data)
    valid_topics, _ = get_valid_topics(topics=topics_no, model="Argument")

    results_overall = dict()
    results_valid = dict()
    i = 1
    while i <= runs:
        name = "run_" + str(i)
        NArgumentModel.get(name, version=3).train(data, test=[])

        _, p_strong_overall, p_both_overall = plot_arg_scoring_eval(NNArgumentModel(findex, name, version=3), topics_no)
        _, p_strong_valid, p_both_valid = plot_arg_scoring_eval(NNArgumentModel(findex, name, version=3), valid_topics)

        result_overall = dict()
        result_overall.setdefault("p_strong", p_strong_overall)
        result_overall.setdefault("p_both", p_both_overall)
        results_overall.setdefault(name, result_overall)

        result_valid = dict()
        result_valid.setdefault("p_strong", p_strong_valid)
        result_valid.setdefault("p_both", p_both_valid)
        results_valid.setdefault(name, result_valid)

        i += 1

    return results_overall, results_valid


def evaluation_nn_argument_topic_split() -> Dict:
    """
    Run Evaluation for NNArgument-Model with specific Topics as Testdata
    :return: Dictionaries with results of runs for all, valid & test topics
    """
    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]
    prep_data = preprocessed_data(findex, topics, train=True)
    data = scale_data(prep_data)
    valid_topics, _ = get_valid_topics(topics=topics_no, model="Argument")
    folds = repeated_kfold(valid_topics=valid_topics)

    results_overall = dict()
    results_valid = dict()
    results_test = dict()
    for i in folds:
        test_topics = folds[i]["test"]
        name = "run_" + str(i)
        NArgumentModel.get(name, version=3).train(data, test=test_topics)

        _, p_strong_overall, p_both_overall = plot_arg_scoring_eval(NNArgumentModel(findex, name, version=3), topics_no)
        _, p_strong_valid, p_both_valid = plot_arg_scoring_eval(NNArgumentModel(findex, name, version=3), valid_topics)
        _, p_strong_test, p_both_test = plot_arg_scoring_eval(NNArgumentModel(findex, name, version=3), test_topics)

        result_overall = dict()
        result_overall.setdefault("p_strong", p_strong_overall)
        result_overall.setdefault("p_both", p_both_overall)
        results_overall.setdefault(name, result_overall)

        result_valid = dict()
        result_valid.setdefault("p_strong", p_strong_valid)
        result_valid.setdefault("p_both", p_both_valid)
        results_valid.setdefault(name, result_valid)

        result_test = dict()
        result_test.setdefault("p_strong", p_strong_test)
        result_test.setdefault("p_both", p_both_test)
        results_test.setdefault(name, result_test)

    return results_overall, results_valid, results_test


def evaluation_nn_stance_image_split(runs: int):
    """
    Run Evaluation for NNStance-Model over all topics with inter train- & test-split
    :param runs: number of runs
    :return: Dictionaries with results of runs
    """
    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]
    prep_data = preprocessed_data(findex, topics, train=True)
    data = scale_data(prep_data)
    valid_topics, _ = get_valid_topics(topics=topics_no, model="Stance")

    results_overall = dict()
    results_valid = dict()
    i = 1
    while i <= runs:
        name = "run_" + str(i)
        NStanceModel.get(name, version=3).train(data, test=[])

        _, accuracy_overall = plot_stance_confusion_eval(NNStanceModel(findex, name, version=3), topics_no)
        _, accuracy_valid = plot_stance_confusion_eval(NNStanceModel(findex, name, version=3), valid_topics)

        result_overall = dict()
        result_overall.setdefault("accuracy", accuracy_overall)
        results_overall.setdefault(name, result_overall)

        result_valid = dict()
        result_valid.setdefault("accuracy", accuracy_valid)
        results_valid.setdefault(name, result_valid)

        i += 1

    return results_overall, results_valid


def evaluation_nn_stance_topic_split() -> Dict:
    """
    Run Evaluation for NNStance-Model with specific Topics as Testdata
    :return: Dictionaries with results of runs for all, valid & test topics
    """
    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]
    prep_data = preprocessed_data(findex, topics, train=True)
    data = scale_data(prep_data)
    valid_topics, _ = get_valid_topics(topics=topics_no, model="Stance")
    folds = repeated_kfold(valid_topics=valid_topics)

    results_overall = dict()
    results_valid = dict()
    results_test = dict()
    for i in folds:
        test_topics = folds[i]["test"]
        name = "run_" + str(i)
        NStanceModel.get(name, version=3).train(data, test=test_topics)

        _, accuracy_overall = plot_stance_confusion_eval(NNStanceModel(findex, name, version=3), topics_no)
        _, accuracy_valid = plot_stance_confusion_eval(NNStanceModel(findex, name, version=3), valid_topics)
        _, accuracy_test = plot_stance_confusion_eval(NNStanceModel(findex, name, version=3), test_topics)

        result_overall = dict()
        result_overall.setdefault("accuracy", accuracy_overall)
        results_overall.setdefault(name, result_overall)

        result_valid = dict()
        result_valid.setdefault("accuracy", accuracy_valid)
        results_valid.setdefault(name, result_valid)

        result_test = dict()
        result_test.setdefault("accuracy", accuracy_test)
        results_test.setdefault(name, result_test)

    return results_overall, results_valid, results_test


def create_results_md_text(results: Dict, model: str, write=False) -> List[str]:
    """
    Create Markdown File with Table of results
    :param write: If True than creating single evaluation md-table
    :param model: Choose "Argument" or "Stance"
    :param results: Dictionary
    :return md text as List
    """
    text = list()
    name = str()

    if model == "Argument":
        name = "results_nn_argument.md"
        text.append("#Results of Runs")
        text.append("##Precision@20")
        text.append("| Run | P_Strong | P_Both |")
        text.append("|---|---|---|")

        average_p_strong = 0
        average_p_both = 0
        for i in results:
            run = str(i)
            run = run.capitalize()
            run = run.replace("_", " ")

            column = "| " + run + " "
            column += "| " + str(results[i]["p_strong"]) + " "
            column += "| " + str(results[i]["p_both"]) + " "
            column += "|"
            text.append(column)

            average_p_strong += results[i]["p_strong"]
            average_p_both += results[i]["p_both"]

        average_p_strong = round(average_p_strong / len(results), 4)
        average_p_both = round(average_p_both / len(results), 4)

        text.append("| Average | " + str(average_p_strong) + " | " + str(average_p_both) + " |")

    if model == "Stance":
        name = "results_nn_stance.md"
        text.append("#Results of Runs")
        text.append("##Accuracy")
        text.append("| Run | Accuracy |")
        text.append("|---|---|")

        average_accuracy = 0
        for i in results:
            run = str(i)
            run = run.capitalize()
            run = run.replace("_", " ")

            column = "| " + run + " "
            column += "| " + str(results[i]["accuracy"]) + " "
            column += "|"
            text.append(column)

            average_accuracy += results[i]["accuracy"]

        average_accuracy = round(average_accuracy / len(results), 4)

        text.append("| Average | " + str(average_accuracy) + " | ")

    if write:
        with open(name, 'w') as f:
            for item in text:
                f.write("%s\n" % item)

    return text


def create_results_md_table(arg_img_overall: List, arg_img_valid: List, arg_topic_overall: List, arg_topic_valid: List,
                            arg_topic_test: List, stance_img_overall: List, stance_img_valid: List,
                            stance_topic_overall: List, stance_topic_valid: List, stance_topic_test: List):
    """
    Create md-table with results of all models
    :param arg_img_overall: List with results of Argument-Model with image split for all topics
    :param arg_img_valid: List with results of Argument-Model with image split for valid topics
    :param arg_topic_overall: List with results of Argument-Model with topic split for all topics
    :param arg_topic_valid: List with results of Argument-Model with topic split for valid topics
    :param arg_topic_test: List with results of Argument-Model with topic split for test topics
    :param stance_img_overall: List with results of Stance-Model with image split for all topics
    :param stance_img_valid: List with results of Stance-Model with image split for valid topics
    :param stance_topic_overall: List with results of Argument-Model with topic split for all topics
    :param stance_topic_valid: List with results of Stance-Model with topic split for valid topics
    :param stance_topic_test: List with results of Stance-Model with topic split for test topics
    """
    text = list()
    text.append("#Evaluation of Runs of the different Models")
    text.append("##1. Evaluation of Argument-Model")

    text.append("###1.1 Evaluation of NeuralNet-Argument Model with datasplit per image")

    text.append("####1.2.1 Evaluation over all labeled topics")
    counter = 0
    for i in arg_img_overall:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####1.2.2 Evaluation over all labeled valid topics")
    counter = 0
    for i in arg_img_valid:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("###1.2 Evaluation of NeuralNet-Argument Model with datasplit per topic")

    text.append("####1.2.1 Evaluation over all labeled topics")
    counter = 0
    for i in arg_topic_overall:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####1.2.2 Evaluation over all labeled valid topics")
    counter = 0
    for i in arg_topic_valid:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####1.2.3 Evaluation over all labeled test topics")
    counter = 0
    for i in arg_topic_test:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("##2. Evaluation of Stance-Model")

    text.append("###2.1 Evaluation of NeuralNet-Stance Model with datasplit per image")

    text.append("####2.1.1 Evaluation over all labeled topics")
    counter = 0
    for i in stance_img_overall:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####2.1.2 Evaluation over all labeled valid topics")
    counter = 0
    for i in stance_img_valid:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("###2.2 Evaluation of NeuralNet-Stance Model with datasplit per topic")

    text.append("####2.2.1 Evaluation over all labeled topics")
    counter = 0
    for i in stance_topic_overall:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####2.2.2 Evaluation over all labeled valid topics")
    counter = 0
    for i in stance_topic_valid:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    text.append("####2.2.3 Evaluation over all labeled test topics")
    counter = 0
    for i in stance_topic_test:
        if counter > 1:
            text.append(i)
        counter += 1
    text.append("\n")

    with open("evaluation_models.md", 'w') as f:
        for item in text:
            f.write("%s\n" % item)


def run_evaluation(runs: int):
    """
    Run Evaluation and Create MD Table
    :param runs: Number of Runs
    """
    results_arg_image_overall, results_arg_image_valid = evaluation_nn_argument_image_split(runs=runs)
    results_arg_topic_overall, results_arg_topic_valid, results_arg_topic_test = evaluation_nn_argument_topic_split()
    results_stance_image_overall, results_stance_image_valid = evaluation_nn_stance_image_split(runs=runs)
    results_stance_topic_overall, results_stance_topic_valid, results_stance_topic_test = \
        evaluation_nn_stance_topic_split()

    results_arg_image_text_overall = create_results_md_text(results=results_arg_image_overall, model="Argument")
    results_arg_image_text_valid = create_results_md_text(results=results_arg_image_valid, model="Argument")

    results_arg_topic_text_overall = create_results_md_text(results=results_arg_topic_overall, model="Argument")
    results_arg_topic_text_valid = create_results_md_text(results=results_arg_topic_valid, model="Argument")
    results_arg_topic_text_test = create_results_md_text(results=results_arg_topic_test, model="Argument")

    results_stance_image_text_overall = create_results_md_text(results=results_stance_image_overall, model="Stance")
    results_stance_image_text_valid = create_results_md_text(results=results_stance_image_valid, model="Stance")

    results_stance_topics_text_overall = create_results_md_text(results=results_stance_topic_overall, model="Stance")
    results_stance_topics_text_valid = create_results_md_text(results=results_stance_topic_valid, model="Stance")
    results_stance_topics_text_test = create_results_md_text(results=results_stance_topic_test, model="Stance")

    create_results_md_table(arg_img_overall=results_arg_image_text_overall,
                            arg_img_valid=results_arg_image_text_valid,
                            arg_topic_overall=results_arg_topic_text_overall,
                            arg_topic_valid=results_arg_topic_text_valid,
                            arg_topic_test=results_arg_topic_text_test,
                            stance_img_overall=results_stance_image_text_overall,
                            stance_img_valid=results_stance_image_text_valid,
                            stance_topic_overall=results_stance_topics_text_overall,
                            stance_topic_valid=results_stance_topics_text_valid,
                            stance_topic_test=results_stance_topics_text_test)

