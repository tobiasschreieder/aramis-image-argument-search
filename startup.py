import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from frontend import start_server
from indexing import Index
from retrieval import RetrievalSystem, DirichletLM


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
    Index.create_index(max_images).save()
    dur = datetime.datetime.now() - then
    log.info('Time for index creation %s', dur)


def main():
    """
    normal program run
    :return:
    """
    log.info('do main stuff')
    index = Index.load(1000)

    system = RetrievalSystem(index, DirichletLM(index))
    # res = system.query('drugs good', 10)
    # log.info(res)
    start_server(system)


if __name__ == '__main__':
    init_logging()
    log = logging.getLogger('startup')
    try:
        main()
    except Exception as e:
        log.error(e, exc_info=True)
        raise e
