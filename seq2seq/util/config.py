"""Configuration file to set up the environment for the rest of the services and
jobs.
"""
import logging
import logging.config
import sys

from os import path
from os import walk
from os import makedirs

import yaml

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def get_root_path():
    """ Get the path to the root directory
    Returns (str):
        Root directory path
    """
    return path.join(path.dirname(path.realpath(__file__)), '../..')


def load_env():
    """ Load .env file into os.environ
    """
    dotenv_path = path.join(get_root_path(), '.env')
    if not path.isfile(dotenv_path):
        logger.error('.env file does not exist. Did you run `bash tools/init_repo.sh?`')
    else:
        load_dotenv(dotenv_path)


def init_logging():
    """ Setup logging configuration using logging.yaml
    """
    root_path = get_root_path()
    log_dump_path = path.join(root_path, 'log')
    if not path.exists(log_dump_path):
        makedirs(log_dump_path)

    logging_path = path.join(root_path, 'logging.yaml')

    # Only configure logging if it has not been configured yet.
    if len(logging.root.handlers) == 0:
        with open(logging_path, 'rt') as file_:
            config = yaml.safe_load(file_.read())
        logging.config.dictConfig(config)


def init():
    """ Required init for any file running in this repository.
    """
    init_logging()
    load_env()
