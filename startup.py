import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler

import cv2

from config import Config
from frontend import start_server
from indexing import StandardTermIndex, FeatureIndex, TopicQueryTermIndex, TopicTermIndex, get_all_topic_indexes, DataEntry
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardStanceModel, StandardArgumentModel


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


def main():
    """
    normal program run
    :return:
    """

    log.info('do main stuff')

    # tq_index = TopicQueryIndex.create_index()
    # tq_index.save()
    tq_index = TopicQueryTermIndex.load()
    topic_indexes = get_all_topic_indexes()
    findex = FeatureIndex.load(23158)

    system = RetrievalSystem(tq_index.prep,
                             topic_model=TopicRankingDirichlet(t_indexes=topic_indexes, tq_index=tq_index),
                             argument_model=StandardArgumentModel(findex),
                             stance_model=StandardStanceModel(findex))

    start_server(system)


if __name__ == '__main__':
    init_logging()
    log = logging.getLogger('startup')
    try:
        Config.get()
        main()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
