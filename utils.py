import logging
import os
import shutil
import tempfile
from logging.handlers import TimedRotatingFileHandler

from joblib import dump, load

from config import Config


class MemmappStore:

    def __init__(self):
        self.temp_folder = tempfile.mkdtemp()

    def store_in_memmap(self, value_to_store, name: str):
        filename = os.path.join(self.temp_folder, 'joblib_{}.mmap'.format(name))
        if os.path.exists(filename):
            os.unlink(filename)
        _ = dump(value_to_store, filename)
        return load(filename, mmap_mode='r+')

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_folder)
        except OSError:
            pass  # this can sometimes fail under Windows


def setup_logger_handler(logger):
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
    logger.addHandler(console)
    logger.addHandler(file_handler)


def get_threading_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if len(logger.handlers) < 1:
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
            fmt='%(asctime)s %(threadName)s %(name)-20s %(funcName)-16.16s %(levelname)-6s %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console)
        logger.addHandler(file_handler)
    return logger
