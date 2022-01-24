import logging
from typing import Dict, List

import pandas as pd
from plotly.express.colors import qualitative

from indexing import TopicQueryTermIndex, get_all_topic_indexes, FeatureIndex, TopicTermIndex, Topic
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardArgumentModel, \
    StandardStanceModel, NNArgumentModel, ArgumentModel
from .analysis_helper import get_relevant_eval, calc_precision_recall, calc_topic_scores, calc_f1
from .configuration import Configuration
from .plot_eval import plot_arg_scoring_eval

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


def main():

    findex = FeatureIndex.load(23158)

    topics_no = [2, 4, 8, 21, 27, 33, 36, 40, 43, 45, 48]
    plot_arg_scoring_eval(NNArgumentModel(findex, 'second_smaller_model'), topics_no).show()
