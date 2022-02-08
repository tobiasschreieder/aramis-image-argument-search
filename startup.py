import argparse
import datetime
import logging
import pathlib
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict

import pandas as pd

from config import Config
from frontend import start_server
from indexing import FeatureIndex, TopicQueryTermIndex, get_all_topic_indexes, \
    Topic, preprocessed_data, scale_data, DataEntry
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardStanceModel, StandardArgumentModel, \
    NNArgumentModel, NNStanceModel

args: Dict[str, Any] = None


def init_logging():
    """
    Method where the root logger is setup
    """

    root = logging.getLogger()
    console = logging.StreamHandler()
    path = Config.get().working_dir.joinpath("logs")
    path.mkdir(parents=True, exist_ok=True)
    file_path = path.joinpath('aramis_imarg_search.log')
    file_handler = TimedRotatingFileHandler(
        filename=str(file_path),
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
    root.debug('Set to debug level')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", default=pathlib.Path('data'), type=pathlib.Path, dest='data_dir')
    parser.add_argument("-o", "--output-dir", default=pathlib.Path('out'), type=pathlib.Path, dest='out_dir')

    parser.add_argument("-w", "--working-dir", default=pathlib.Path('working'), type=pathlib.Path, dest='work_dir')
    parser.add_argument("-cfg", "--config", default=pathlib.Path('config.json'), type=pathlib.Path, dest='config')
    parser.add_argument("-f", "--image_format", action='store_true', dest='image_format')

    parser.add_argument('-c', '--count_images', action='store_true', dest='count_ids')
    parser.add_argument('-idx', '--indexing', action='store_true', dest='indexing')
    parser.add_argument('-tidx', '--test-indexing', action='store_true', dest='test_indexing')
    parser.add_argument('-njobs', '--number-jobs', type=int, dest='n_jobs', default=-2)
    parser.add_argument('-qrel', '--qrel', action='store_true', dest='qrel')

    parser.add_argument('-web', '--web-frontend', action='store_true', dest='frontend')
    parser.add_argument('-p', '--port', type=int, dest='port', default=5000)
    parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0')

    global args
    args = parser.parse_args()
    args = vars(args)

    if 'config' in args.keys():
        Config._save_path = args['config']

    cfg = Config.get()
    if 'data_dir' in args.keys():
        cfg.data_dir = args['data_dir']
    if 'out_dir' in args.keys():
        cfg.output_dir = args['out_dir']
    if 'work_dir' in args.keys():
        cfg.working_dir = args['work_dir']
    if 'image_format' in args.keys():
        cfg.data_image_format = args['image_format']

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()


def handle_args():
    if args['count_ids']:
        log.info('Found %s images in data.', len(DataEntry.get_image_ids()))
        sys.exit(0)

    if args['test_indexing']:
        log.info('Start term index creation for %s images', 5)
        Topic.load_all()[0].get_image_ids()
        then = datetime.datetime.now()
        TopicQueryTermIndex.create_index(max_images=5, n_jobs=args['n_jobs']).save()
        get_all_topic_indexes(n_jobs=args['n_jobs'], max_images=5)
        log.info('Start feature index creation for %s images', 5)
        fidx = FeatureIndex.create_index(max_images=5, n_jobs=args['n_jobs'])
        fidx.save()
        log.info('Precalculate data for retrieval process')
        preprocessed_data(fidx, Topic.load_all())
        dur = datetime.datetime.now() - then
        log.info('Time for index creation %s', dur)
        sys.exit(0)

    if args['indexing']:
        max_id = len(DataEntry.get_image_ids())
        index_creation(max_id, n_jobs=args['n_jobs'])
        sys.exit(0)

    if args['qrel']:
        log.info('Start qrel scoring')
        qrel_scoring()
        sys.exit(0)

    if args['frontend']:
        log.info('Start flask frontend')
        start_server(get_rs(1, 'model_1', 'model_1'), host=args['host'], port=args['port'])
        sys.exit(0)

    main()


def get_rs(tw: float, model_arg: str = None, model_stance: str = None) -> RetrievalSystem:
    tq_index = TopicQueryTermIndex.load()
    topic_indexes = get_all_topic_indexes()
    fidx = FeatureIndex.load(-1)
    if model_arg is None:
        arg_m = StandardArgumentModel(fidx)
    else:
        arg_m = NNArgumentModel(fidx, model_arg)
    if model_stance is None:
        stance_m = StandardStanceModel(fidx)
    else:
        stance_m = NNStanceModel(fidx, model_stance)
    return RetrievalSystem(
        tq_index.prep,
        topic_model=TopicRankingDirichlet(
            t_indexes=topic_indexes, tq_index=tq_index, alpha=1000, tq_alpha=1000
        ),
        argument_model=arg_m,
        stance_model=stance_m,
        topic_weight=tw,
    )


def qrel_scoring():
    log.info('Load indices')
    rs = get_rs(1, 'model_1', 'model_1')
    data = []
    log.info('loading done, start scoring')
    for topic in Topic.load_all():
        result_p, result_c = rs.query(topic.title, top_k=10, topic=topic)
        for i, r in enumerate(result_p):
            data.append([topic.number, 'PRO', r[0], i, r[1], 'NN_model1-w1'])
        for i, r in enumerate(result_c):
            data.append([topic.number, 'CON', r[0], i, r[1], 'NN_model1-w1'])
    df = pd.DataFrame(data, columns=['topic', 'stance', 'image_id', 'rank', 'score', 'method'])
    file_path = Config.get().output_dir.joinpath('run.txt')
    df.to_csv(file_path, sep=' ', header=False, index=False)
    log.info('scoring saved under %s', file_path)


def index_creation(max_images: int, n_jobs: int = -2) -> None:
    log.info('Start term index creation for %s images', max_images)
    then = datetime.datetime.now()
    TopicQueryTermIndex.create_index(max_images=max_images, n_jobs=n_jobs).save()
    get_all_topic_indexes(n_jobs=n_jobs)
    log.info('Start feature index creation for %s images', max_images)
    fidx = FeatureIndex.create_index(max_images=max_images, n_jobs=n_jobs)
    fidx.save()
    log.info('Precalculate data for retrieval process')
    preprocessed_data(fidx, Topic.load_all())
    dur = datetime.datetime.now() - then
    log.info('Time for index creation %s', dur)


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
    #
    # prep_data = preprocessed_data(findex, [Topic.get(31)], train=True)
    # data = scale_data(prep_data)

    # NArgumentModel.get('test_final_2', version=3).train(data, test=eval_topics)
    # NStanceModel.get('test_final_2', version=3).train(data, test=eval_topics)

    # analysis_main(model_name='test_final_2', topics_no=topics_no, version=3)
    # analysis_main(model_name='test_final_2', topics_no=eval_topics, version=3)
    # retrieval_system_analysis.eval_nn_model()
    # retrieval_system_analysis.eval_standard_model()
    # retrieval_system_analysis.eval_baseline()

    # analyse_network_features_arg(data)
    # analyse_network_features_stance(data)


if __name__ == '__main__':
    parse_args()
    init_logging()
    log = logging.getLogger('startup')
    try:
        handle_args()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
