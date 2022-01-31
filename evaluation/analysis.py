import logging

from plotly.express.colors import qualitative

from indexing import FeatureIndex
from retrieval import StandardArgumentModel, \
    StandardStanceModel, NNArgumentModel, NNStanceModel
from .plot_eval import plot_arg_scoring_eval, plot_stance_confusion

log = logging.getLogger('analysis')


plotly_color = qualitative.Plotly


def main(model_name: str, topics_no: list, version: int):

    findex = FeatureIndex.load(23158)

    # plot_arg_scoring_eval(StandardArgumentModel(findex), topics_no).show()
    # plot_arg_scoring_eval(StandardStanceModel(findex), topics_no).show()
    # plot_stance_confusion(StandardStanceModel(findex), topics_no).show()

    # plot_arg_scoring_eval(NNArgumentModel(findex, model_name, version=version), topics_no).show()
    plot_stance_confusion(NNStanceModel(findex, model_name, version=version), topics_no).show()
