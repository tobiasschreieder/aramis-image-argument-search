import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from config import Config
from evaluation.analysis import main as analysis_main
from evaluation.feature_analysis import analyse_network_features_arg, analyse_network_features_stance
from frontend import start_server
from indexing import StandardTermIndex, FeatureIndex, TopicQueryTermIndex, get_all_topic_indexes, \
    Topic, NStanceModel, preprocessed_data, scale_data, NArgumentModel
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardStanceModel, StandardArgumentModel
from evaluation import retrieval_system_analysis


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

    # eval_topics = [9, 27, 31, 33]
    # skip_topics = [15, 31, 36, 37, 43, 45, 48]
    # rest_topics = [1, 2, 4, 8, 10, 20, 21, 22, 40, 47]

    # findex = FeatureIndex.load(23158)
    # topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    # topics = [Topic.get(t) for t in topics_no]

    # prep_data = preprocessed_data(findex, topics, train=True)
    # data = scale_data(prep_data)

    # NArgumentModel.get('test_final_2', version=3).train(data, test=eval_topics)
    # NStanceModel.get('test_final_2', version=3).train(data, test=eval_topics)

    # analysis_main(model_name='test_final_2', topics_no=topics_no, version=3)
    # analysis_main(model_name='test_final_2', topics_no=eval_topics, version=3)
    # retrieval_system_analysis.eval_nn_model()
    # retrieval_system_analysis.eval_standard_model()
    retrieval_system_analysis.eval_baseline()

    # analyse_network_features_arg(data)
    # analyse_network_features_stance(data)


if __name__ == '__main__':
    init_logging()
    log = logging.getLogger('startup')
    try:
        Config.get()
        main()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
