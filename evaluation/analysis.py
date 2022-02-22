import logging

from indexing import FeatureIndex
from retrieval import NNArgumentModel, NNStanceModel
from .plot_eval import plot_arg_scoring_eval, plot_stance_confusion_eval

log = logging.getLogger('analysis')


def main(model_name: str, topics_no: list, version: int):

    findex = FeatureIndex.load(23158)

    # plot_arg_scoring_eval(StandardArgumentModel(findex), topics_no)[0].show()
    # plot_arg_scoring_eval(StandardStanceModel(findex), topics_no)[0].show()
    # plot_stance_confusion_eval(StandardStanceModel(findex), topics_no)[0].show()

    plot_arg_scoring_eval(NNArgumentModel(findex, model_name, version=version), topics_no)[0].show()
    plot_stance_confusion_eval(NNStanceModel(findex, model_name, version=version), topics_no)[0].show()
