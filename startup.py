import argparse
import datetime
import logging
import pathlib
import sys
from typing import Any, Dict

import pandas as pd

from config import Config
from evaluation.analysis import main as analysis_main
from frontend import start_server
from indexing import FeatureIndex, TopicQueryTermIndex, get_all_topic_indexes, \
    Topic, preprocessed_data, DataEntry
from retrieval import RetrievalSystem, TopicRankingDirichlet, StandardStanceModel, StandardArgumentModel, \
    NNArgumentModel, NNStanceModel
from utils import setup_logger_handler

args: Dict[str, Any] = None


def init_logging():
    """
    Method where the root logger is setup
    """

    root = logging.getLogger()
    setup_logger_handler(root)
    root.setLevel(logging.DEBUG)

    root.info('Logging initialised')
    root.debug('Set to debug level')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", default=pathlib.Path('data'), type=pathlib.Path,
                        dest='data_dir', help='Path to input directory.')
    parser.add_argument("-o", "--output-dir", default=pathlib.Path('out'), type=pathlib.Path,
                        dest='out_dir', help='Path to output directory.')

    parser.add_argument("-w", "--working-dir", default=pathlib.Path('working'), type=pathlib.Path,
                        dest='work_dir', help='Path to working directory. (Location of index/neural net models)')
    parser.add_argument("-cfg", "--config", default=pathlib.Path('config.json'), type=pathlib.Path,
                        dest='config', help='Path to config.json file.')
    parser.add_argument("-f", "--image_format", action='store_true',
                        dest='image_format', help='Specifies format of input data. See README for definition.')

    parser.add_argument('-c', '--count_images', action='store_true',
                        dest='count_ids', help='Performs a count of found images in input.')
    parser.add_argument('-idx', '--indexing', action='store_true', dest='indexing',
                        help='Calculate the index on the given input.')
    parser.add_argument('-tidx', '--test-indexing', action='store_true', dest='test_indexing',
                        help='Perform a small indexing run with only 5 images to test the indexing.')
    parser.add_argument('-njobs', '--number-jobs', type=int, dest='n_jobs', default=-1,
                        help='Number of processors to use in parallel indexing process. -1 = all Processors,'
                             ' -2 = all processors but one')
    parser.add_argument('-qrel', '--qrel', action='store_true', dest='qrel',
                        help='Perform a retrieval run over all topics and create run.txt')
    parser.add_argument('-mtag', '--method_tag', type=str, dest='method_tag', default='aramis#standard#standard#w0.5',
                        help='Retrieval method tag for retrival run. '
                             'Format: "aramis#{ArgumentModel}#{StanceModel}#w{topic_weight}"'
                             'ArgumentModel/StanceModel: standard or NN_{model_name}, topic_weight: float in [0,1]')

    parser.add_argument('-web', '--website', action='store_true', dest='frontend',
                        help='Start flask web server.')
    parser.add_argument('-p', '--port', type=int, dest='port', default=5000,
                        help='Port for web server.')
    parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0',
                        help='Host address for web server.')

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
        log.info('Start qrel scoring with method tag %s', args['method_tag'])
        qrel_scoring(args['method_tag'])
        sys.exit(0)

    if args['frontend']:
        log.info('Start flask frontend with method tag %s', args['method_tag'])
        start_server(parse_method_tag(args['method_tag']), host=args['host'], port=args['port'])
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


def parse_method_tag(method_tag: str) -> RetrievalSystem:
    """
    Create RetrievalSystem for given method tag.
    Method tag should have format 'aramis#{ArgumentModel}#{StanceModel}#w{topic_weight}' where
     - ArgumentModel is 'standard' or 'NN-{model_name}'
     - StanceModel is 'standard' or 'NN-{model_name}'
     - topic_weight is float in [0,1]
    :param method_tag: string to parse
    :return: RetrievalSystem for parsed method tag
    :raise ValueError: if method tag is faulty
    """
    split = method_tag.split('#')
    if len(split) == 4 and split[0] == 'aramis':
        # ArgumentModel
        if split[1] == 'standard':
            arg_model = None
        elif split[1][:3] == 'NN_':
            arg_model = split[1][3:]
        else:
            raise ValueError('ArgumentModel {} not found'.format(split[1]))

        # StanceModel
        if split[2] == 'standard':
            stance_model = None
        elif split[2][:3] == 'NN_':
            stance_model = split[2][3:]
        else:
            raise ValueError('StanceModel {} not found'.format(split[2]))

        # TopicWeight
        try:
            t_w = float(split[3].strip()[1:])
            if not (0 <= t_w <= 1):
                raise ValueError
        except ValueError:
            raise ValueError('Topic weight {} is not a number in [0,1]'.format(split[3]))

        return get_rs(t_w, arg_model, stance_model)
    raise ValueError('Method tag "%s" is not correctly formatted. '
                     'Correct format: "aramis#{ArgumentModel}#{StanceModel}#w{topic_weight}"')


def qrel_scoring(method_tag: str):
    log.info('Load indices')
    rs = parse_method_tag(method_tag)
    data = []
    log.info('loading done, start scoring')
    for topic in Topic.load_all():
        result_p, result_c = rs.query(topic.title, top_k=10, topic=topic)
        for i, r in enumerate(result_p):
            data.append([topic.number, 'PRO', r[0], i+1, round(r[1], 6), method_tag])
        for i, r in enumerate(result_c):
            data.append([topic.number, 'CON', r[0], i+1, round(r[1], 6), method_tag])
    df = pd.DataFrame(data, columns=['topic', 'stance', 'image_id', 'rank', 'score', 'method'])
    file_path = Config.get().output_dir.joinpath('run.txt')
    df.to_csv(file_path, sep=' ', header=False, index=False)
    log.info('scoring saved under %s', file_path)


def index_creation(max_images: int, n_jobs: int = -2) -> None:
    log.info('Start term index creation for %s images', max_images)
    then = datetime.datetime.now()
    TopicQueryTermIndex.create_index(max_images=max_images, n_jobs=n_jobs).save()
    get_all_topic_indexes(n_jobs=n_jobs, force_create=True)
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

    # eval_topics = [9, 27, 31, 33]
    # skip_topics = [15, 31, 36, 37, 43, 45, 48]
    # rest_topics = [1, 2, 4, 8, 10, 20, 21, 22, 40, 47]

    findex = FeatureIndex.load(23158)
    topics_no = [1, 2, 4, 8, 9, 10, 15, 20, 21, 22, 27, 31, 33, 36, 37, 40, 43, 45, 47, 48]
    topics = [Topic.get(t) for t in topics_no]

    # prep_data = preprocessed_data(findex, topics, train=True)
    # data = scale_data(prep_data)

    # NArgumentModel.get('model_2', version=3).train(data, test=[])
    # NStanceModel.get('model_2', version=3).train(data, test=[])

    analysis_main(model_name='model_1', topics_no=topics_no, version=3)

    # retrieval_system_analysis.eval_nn_model()
    # retrieval_system_analysis.eval_standard_model()
    # retrieval_system_analysis.eval_baseline()

    # analyse_network_features_arg(data)
    # analyse_network_features_stance(data)

    # crossvalidation.run_evaluation(runs=10)


if __name__ == '__main__':
    parse_args()
    init_logging()
    log = logging.getLogger('startup')
    try:
        handle_args()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
