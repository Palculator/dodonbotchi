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
from dodonbotchi.config import ensure_config
from dodonbotchi.exy import EXY
from dodonbotchi.mame import RECORDING_FILE
from dodonbotchi.util import ensure_directories

TIMESTAMP = datetime.now().isoformat().replace(':', '_')
DEF_LOG = 'log'
DEF_CFG = 'dodonbotchi.cfg'
LOG_FIL = 'dodonbotchi_{}.log'.format(TIMESTAMP)


def log_exception(extype, value, trace):
    """
    Hook to log uncaught exceptions to the logging framework. Register this as
    the excepthook with `sys.excepthook = log_exception`.
    """
    log.exception('Uncaught exception: ', exc_info=(extype, value, trace))


def setup_logging(log_dir):
    """
    Sets up the logging framework to log to the given log_file and to STDOUT.
    If the path to the log_file does not exist, directories for it will be
    created.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, LOG_FIL)

    file_handler = log.FileHandler(log_file, 'w', 'utf-8')
    term_handler = log.StreamHandler()
    handlers = [term_handler, file_handler]

    fmt = '%(asctime)s %(levelname)-8s %(message)s'

    log.basicConfig(handlers=handlers, format=fmt, level=log.DEBUG)

    sys.excepthook = log_exception

    log.info('Started DoDonBotchi logging to: %s', log_file)


@click.group()
@click.option('--log-dir', type=click.Path(file_okay=False), default=DEF_LOG)
@click.option('--cfg-file', type=click.Path(dir_okay=False), default=DEF_CFG)
def cli(log_dir=None, cfg_file=None):
    """
    Click group that ensures at least log and configuration files are present,
    since the rest of the application uses those.
    """
    setup_logging(log_dir)
    ensure_config(cfg_file)


@cli.command()
@click.argument('exy-dir', type=click.Path(file_okay=False))
def train(exy_dir):
    """
    Trains a neural net writing trial runs, captures, progress information, and
    brain states to the specified output folder until the user interrupts
    training.
    """
    exy = EXY(exy_dir)
    exy.train()


@cli.command()
@click.argument('exy-dir', type=click.Path(file_okay=False))
@click.argument('video-file', type=click.Path(dir_okay=False))
def render_all(exy_dir, video_file):
    """
    Renders each episode of the EXY training at the given EXY dir into one
    (probably very long) video file. Might take ages.
    """
    exy_dir = os.path.abspath(exy_dir)
    exy = EXY(exy_dir)

    video_file = os.path.abspath(video_file)
    inp_dir = exy.get_episodes_dir()
    snp_dir = os.path.join(exy_dir, '.tmp')
    ensure_directories(snp_dir)

    avi_files = []
    for episode in sorted(os.listdir(inp_dir)):
        ep_dir = os.path.join(inp_dir, episode)
        if not os.path.isdir(ep_dir):
            continue

        inp_file = os.path.join(episode, RECORDING_FILE)

        avi_file = episode + '.avi'
        mame.render_avi(inp_file, avi_file, inp_dir, snp_dir)

        avi_file = os.path.join(snp_dir, avi_file)
        avi_files.append(avi_file)

    list_file = os.path.join(snp_dir, 'avis.ls')
    with open(list_file, 'w') as out_file:
        for avi_file in avi_files:
            line = 'file \'{}\'\n'.format(avi_file)
            out_file.write(line)

    call = ['ffmpeg', '-safe', '0', '-f', 'concat', '-i', list_file,
            '-c:v', 'libx264', '-crf', '11', '-c:a', 'aac', '-b:a', '192k',
            '-vf', 'scale=960:1280:flags=lanczos',
            video_file]
    subprocess.call(call)
    shutil.rmtree(snp_dir)


@cli.command()
@click.option('--amount', default=10)
@click.argument('exy-dir', type=click.Path(file_okay=False))
@click.argument('video-dir', type=click.Path(file_okay=False))
def render_leaderboard(amount, exy_dir, video_dir):
    """
    Renders the top players in the given EXY directory's leaderboard. The given
    amount controls how many entries get rendered, starting from highest score
    to lowest. Videos will be temporarily stored in the EXY directory but
    finally rendered to the given video directory.
    """
    exy_dir = os.path.abspath(exy_dir)
    exy = EXY(exy_dir)
    exy.load_properties()

    video_dir = os.path.abspath(video_dir)
    inp_dir = exy.get_episodes_dir()
    snp_dir = os.path.join(exy_dir, '.tmp')
    ensure_directories(video_dir, snp_dir)

    leaderboard = exy.leaderboard

    if len(leaderboard) > amount:
        leaderboard = leaderboard[:amount]

    for idx, entry in enumerate(leaderboard):
        inp_file = os.path.join(entry['ep'], RECORDING_FILE)
        score = entry['score']

        video_file = '{:03} - {} - {:09}'.format(idx + 1, entry['ep'], score)
        avi_file = video_file + '.avi'
        mp4_file = video_file + '.mp4'
        mp4_file = os.path.join(video_dir, mp4_file)

        mame.render_avi(inp_file, avi_file, inp_dir, snp_dir)

        avi_file = os.path.join(snp_dir, avi_file)
        call = ['ffmpeg', '-i', avi_file,
                '-c:v', 'libx264', '-crf', '11',
                '-c:a', 'aac', '-b:a', '192k',
                '-vf', 'scale=960:1280:flags=lanczos',
                mp4_file]
        subprocess.call(call)

    shutil.rmtree(snp_dir)


if __name__ == '__main__':
    cli()
