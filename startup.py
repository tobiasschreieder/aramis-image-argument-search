import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from indexing.index import Index


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


def main():
    """
    normal programm run
    :return:
    """
    logger.info('do main stuff')

    then = datetime.datetime.now()
    Index.create_index(100).save()
    dur = datetime.datetime.now() - then
    # Index.load(10)
    logger.info(dur)


if __name__ == '__main__':
    init_logging()
    logger = logging.getLogger('startup')
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e
