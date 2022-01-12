import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from config import Config
from frontend import start_server
from indexing import StandardTermIndex, FeatureIndex, TopicQueryTermIndex, TopicTermIndex, get_all_topic_indexes, \
    DataEntry, Topic
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardStanceModel, StandardArgumentModel
from evaluation import save_eval, Argumentative, Stance, get_eval, has_eval
from indexing.feature import html_preprocessing
import numpy as np
from evaluation.analysis import main as analysis_main


def init_logging():
    """
    Method where the root logger is setup
    """
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    root = logging.getLogger()
    console = logging.StreamHandler()
    file_handler = TimedRotatingFileHandler(
        filename="logs/aramis_imarg_search.log",
        utc=True,
        when='midnight'
    )
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-20s %(funcName)-16.16s %(levelname)-6s %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)

    root.setLevel(logging.DEBUG)

    root.info('Logging initialised')


def index_creation(max_images: int) -> None:
    log.info('Start index creation')
    then = datetime.datetime.now()
    StandardTermIndex.create_index(max_images).save()
    dur = datetime.datetime.now() - then
    log.info('Time for index creation %s', dur)


def start_flask() -> None:
    tq_index = TopicQueryTermIndex.load()
    topic_indexes = get_all_topic_indexes()
    findex = FeatureIndex.load(23158)

    system = RetrievalSystem(tq_index.prep,
                             topic_model=TopicRankingDirichlet(t_indexes=topic_indexes, tq_index=tq_index),
                             argument_model=StandardArgumentModel(findex),
                             stance_model=StandardStanceModel(findex))

    start_server(system)


def main():
    """
    normal program run
    :return:
    """

    log.info('do main stuff')

    # start_flask()
    # findex = FeatureIndex.create_index(10, n_jobs=6)
    # findex.calculate_sentiment_score_v2(n_jobs=6)
    # findex.save()
    # start_server(None)

    # data = html_preprocessing.html_test()
    # np.save('data12', data)

    analysis_main()


if __name__ == '__main__':
    init_logging()
    log = logging.getLogger('startup')
    try:
        Config.get()
        main()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
