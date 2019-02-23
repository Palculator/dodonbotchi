"""
This module serves as the main entry point to DoDonBotchi. It takes care of
setting up logging, reading global configuration, and defines a command line
interface to invoke DoDonBotchi's functions. Start this script with `--help` to
get information on said CLI.
"""

import json
import logging as log
import os
import shutil
import subprocess
import sys

from datetime import datetime

import click

from dodonbotchi import mame
from dodonbotchi import exy
from dodonbotchi.config import ensure_config
from dodonbotchi.mame import RECORDING_FILE
from dodonbotchi.util import ensure_directories

DEF_LOG = 'dodonbotchi.log'
DEF_CFG = 'dodonbotchi.cfg'


def log_exception(extype, value, trace):
    """
    Hook to log uncaught exceptions to the logging framework. Register this as
    the excepthook with `sys.excepthook = log_exception`.
    """
    log.exception('Uncaught exception: ', exc_info=(extype, value, trace))


def setup_logging(log_file, no_file=False):
    """
    Sets up the logging framework to log to the given log_file and to STDOUT.
    If the path to the log_file does not exist, directories for it will be
    created.
    """
    if os.path.exists(log_file):
        backup = f'{log_file}.1'
        shutil.move(log_file, backup)

    term_handler = log.StreamHandler()
    handlers = [term_handler]
    if not no_file:
        file_handler = log.FileHandler(log_file, 'w', 'utf-8')
        handlers.append(file_handler)

    fmt = '%(asctime)s %(levelname)-8s %(message)s'

    log.basicConfig(handlers=handlers, format=fmt, level=log.INFO)

    sys.excepthook = log_exception

    log.info('Started DoDonBotchi logging to: %s', log_file)


@click.group()
@click.option('--log-file', type=click.Path(dir_okay=False), default=DEF_LOG)
@click.option('--cfg-file', type=click.Path(dir_okay=False), default=DEF_CFG)
@click.option('--no-file', is_flag=True)
def cli(log_file=None, cfg_file=None, no_file=False):
    """
    Click group that ensures at least log and configuration files are present,
    since the rest of the application uses those.
    """
    setup_logging(log_file, no_file)
    ensure_config(cfg_file)


@cli.command()
@click.argument('cwd', type=click.Path(file_okay=False))
def progression(cwd):
    exy.evolve(cwd)


@cli.command()
@click.argument('cwd', type=click.Path(file_okay=False))
@click.argument('recording')
def replay(cwd, recording):
    exy.replay(cwd, recording)


if __name__ == '__main__':
    cli()
