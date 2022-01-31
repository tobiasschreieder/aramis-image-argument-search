import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from config import Config
from evaluation.analysis import main as analysis_main
from evaluation.feature_analysis import analyse_network_features_arg, analyse_network_features_stance
from frontend import start_server
from indexing import StandardTermIndex, FeatureIndex, TopicQueryTermIndex, get_all_topic_indexes, \
    Topic, NStanceModel, preprocessed_data, scale_data
from indexing.neural_net.arg_network import NArgumentModel
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

    # findex = FeatureIndex.create_index()
    # findex.save()

    # start_server(None)

    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]

    data = scale_data(preprocessed_data(findex, topics, train=True))

    # NArgumentModel.get('test_final', version=3).train(data, test=[27, 31, 33])
    NStanceModel.get('test_1', version=3).train(data, test=[27, 31, 33])

    analysis_main(model_name='test_final', topics_no=topics_no, version=3)
    # rs_analysis_main()

    # analyse_network_features_arg(data)
    analyse_network_features_stance(data)


if __name__ == '__main__':
    init_logging()
    log = logging.getLogger('startup')
    try:
        Config.get()
        main()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
