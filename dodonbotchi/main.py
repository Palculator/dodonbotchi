"""
This module serves as the main entry point to DoDonBotchi. It takes care of
setting up logging, reading global configuration, and defines a command line
interface to invoke DoDonBotchi's functions. Start this script with `--help` to
get information on said CLI.
"""

import logging as log
import os
import sys

from datetime import datetime

import click

from dodonbotchi.config import ensure_config
from dodonbotchi.exy import EXY

TIMESTAMP = datetime.now().isoformat().replace(':', '_')
DEF_LOG = 'log/dodonbotchi_{}.log'.format(TIMESTAMP)
DEF_CFG = 'dodonbotchi.cfg'


def log_exception(extype, value, trace):
    """
    Hook to log uncaught exceptions to the logging framework. Register this as
    the excepthook with `sys.excepthook = log_exception`.
    """
    log.exception('Uncaught exception: ', exc_info=(extype, value, trace))


def setup_logging(log_file):
    """
    Sets up the logging framework to log to the given log_file and to STDOUT.
    If the path to the log_file does not exist, directories for it will be
    created.
    """
    log_dir, _ = os.path.split(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = log.FileHandler(log_file, 'w', 'utf-8')
    term_handler = log.StreamHandler()
    handlers = [term_handler, file_handler]

    fmt = '%(asctime)s %(levelname)-8s %(message)s'

    log.basicConfig(handlers=handlers, format=fmt, level=log.DEBUG)

    sys.excepthook = log_exception

    log.info('Started DoDonBotchi logging to: %s', log_file)


@click.group()
@click.option('--log-file', type=click.Path(dir_okay=False), default=DEF_LOG)
@click.option('--cfg-file', type=click.Path(dir_okay=False), default=DEF_CFG)
def cli(log_file=None, cfg_file=None):
    """
    Click group that ensures at least log and configuration files are present,
    since the rest of the application uses those.
    """
    setup_logging(log_file)
    ensure_config(cfg_file)


@cli.command()
@click.option('--seed', default=32512)
@click.argument('output', type=click.Path(file_okay=False))
def train(seed, output):
    """
    Trains a neural net writing trial runs, captures, progress information, and
    brain states to the specified output folder until the user interrupts
    training.
    """
    exy = EXY(output)
    exy.train(seed)


if __name__ == '__main__':
    cli()
