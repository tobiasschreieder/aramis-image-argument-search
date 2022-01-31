import logging
from typing import Dict, List

import pandas as pd
from plotly.express.colors import qualitative

from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardArgumentModel, \
    StandardStanceModel, NNArgumentModel, ArgumentModel, NNStanceModel
from .analysis_helper import get_relevant_eval, calc_precision_recall, calc_topic_scores, calc_f1
from .configuration import Configuration
from .plot_eval import plot_arg_scoring_eval, plot_stance_scoring_eval, plot_stance_confusion

log = logging.getLogger('analysis')


plotly_color = qualitative.Plotly
# RetrievalSystem:
#   - topic_weight
#   - argument_weight
#   - prefetch_top_k
#
# TopicRankingDirichlet
#   - topic query alpha
#   - alpha

# Needed:
#   metrik retrieval result -> score
#       -> precision/recall - f1-funktion (@k)

# function configuration in f1 out
# 3-fold (bootstrap) over topics (choose rdm 80% topics eval with rest) (3/1)

# 1. f1 system eval = avg over all topics (5,5)
# calc eval for each parameter range, save best arg
# best cfg = all best parameters


def main(model_name: str, topics_no: list):

    findex = FeatureIndex.load(23158)

    # plot_arg_scoring_eval(NNArgumentModel(findex, model_name, version=3), topics_no).show()
    # plot_stance_scoring_eval(NNStanceModel(findex, model_name_stance), topics_no).show()
    plot_stance_confusion(NNStanceModel(findex, model_name, version=3), topics_no).show()
