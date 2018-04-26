import logging as log
import os
import sys
from datetime import datetime

import click

from dodonbotchi.config import ensure_config, cfg
from dodonbotchi.mamer import run_mame

DEFAULT_LOG = 'log/dodonbotchi_{}.log'.format(datetime.now().isoformat())
DEFAULT_CFG = 'dodonbotchi.cfg'


def log_exception(extype, value, trace):
    log.exception('Uncaught exception: ', exc_info=(extype, value, trace))


def setup_logging(log_file):
    log_dir, _ = os.path.split(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = log.FileHandler(log_file, 'w', 'utf-8')
    term_handler = log.StreamHandler()
    handlers = [term_handler, file_handler]

    format = '%(asctime)s %(levelname)-8s %(message)s'

    log.basicConfig(handlers=handlers, format=format, level=log.DEBUG)

    sys.excepthook = log_exception

    log.info('Started DoDonBotchi logging to: %s', log_file)


@click.group()
@click.option('--log-file', type=click.Path(), default=DEFAULT_LOG)
@click.option('--cfg-file', type=click.Path(), default=DEFAULT_CFG)
def cli(log_file, cfg_file):
    setup_logging(log_file)
    ensure_config(cfg_file)


@cli.command()
def run():
    run_mame()


if __name__ == '__main__':
    cli()
