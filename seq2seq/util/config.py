"""Configuration file to set up the environment for the rest of the services and
jobs.
"""
import logging
import logging.config

from os import path
from os import makedirs

import yaml

logger = logging.getLogger(__name__)


def get_root_path():
    """ Get the path to the root directory
    Returns (str):
        Root directory path
    """
    return path.join(path.dirname(path.realpath(__file__)), '../..')


def init_logging():
    """ Setup logging configuration using logging.yaml
    """
    if len(logging.root.handlers) == 0:
        root_path = get_root_path()
        log_dump_path = path.join(root_path, 'log')
        if not path.exists(log_dump_path):
            makedirs(log_dump_path)

        logging_path = path.join(root_path, 'logging.yaml')

        # Only configure logging if it has not been configured yet.
        with open(logging_path, 'rt') as file_:
            config = yaml.safe_load(file_.read())
        logging.config.dictConfig(config)
