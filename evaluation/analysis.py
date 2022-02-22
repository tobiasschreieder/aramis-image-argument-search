import logging

from indexing import FeatureIndex, NArgumentModel, NStanceModel, scale_data, preprocessed_data, Topic
from retrieval import NNArgumentModel, NNStanceModel
from .plot_eval import plot_arg_scoring_eval, plot_stance_confusion_eval

log = logging.getLogger('analysis')


def find_best_models():
    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]

    prep_data = preprocessed_data(findex, topics, train=True)
    data = scale_data(prep_data)

    best_arg_model_name = 'none'
    best_stance_model_name = 'none'
    best_p = [0, 0]
    best_acc = 0
    best_arg = 0
    best_stance = 0
    for i in range(50):
        NArgumentModel.get(f'find_{best_arg}', version=3).train(data, test=[])
        NStanceModel.get(f'find_{best_stance}', version=3).train(data, test=[])

        _, pstrong, pboth = plot_arg_scoring_eval(NNArgumentModel(findex, f'find_{best_arg}', version=3), topics_no)
        if pboth > best_p[1]:
            best_arg_model_name = f'find_{best_arg}'
            best_arg += 1
            best_p = [pstrong, pboth]

        _, acc = plot_stance_confusion_eval(NNStanceModel(findex, f'find_{best_stance}', version=3), topics_no)
        if acc > best_acc:
            best_stance_model_name = f'find_{best_stance}'
            best_stance += 1
            best_acc = acc
    log.info('Found best ArgumentModel %s with PStrong@20: %S, PBoth@20: %s', best_arg_model_name, best_p[0], best_p[1])
    log.info('Found best StanceModel %s with Accuracy: %S', best_stance_model_name, best_acc)


def main(model_name: str, topics_no: list, version: int):
    findex = FeatureIndex.load(23158)

    # plot_arg_scoring_eval(StandardArgumentModel(findex), topics_no)[0].show()
    # plot_arg_scoring_eval(StandardStanceModel(findex), topics_no)[0].show()
    # plot_stance_confusion_eval(StandardStanceModel(findex), topics_no)[0].show()

    plot_arg_scoring_eval(NNArgumentModel(findex, model_name, version=version), topics_no)[0].show()
    plot_stance_confusion_eval(NNStanceModel(findex, model_name, version=version), topics_no)[0].show()
