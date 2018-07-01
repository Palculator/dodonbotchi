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
from dodonbotchi import ddon
from dodonbotchi.config import ensure_config
from dodonbotchi.exy import EXY
from dodonbotchi.mame import RECORDING_FILE
from dodonbotchi.util import ensure_directories, generate_now_serial_number

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

    log.basicConfig(handlers=handlers, format=fmt, level=log.DEBUG)

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


@cli.group()
def rl():
    pass


@rl.command()
@click.option('--agent')
@click.argument('exy-dir', type=click.Path(file_okay=False))
def train(agent, exy_dir):
    """
    Trains a neural net writing trial runs, captures, progress information, and
    brain states to the specified output folder until the user interrupts
    training.
    """
    exy = EXY(exy_dir, agent)
    exy.train()


@rl.command()
@click.option('--amount', default=10)
@click.option('--agent')
@click.argument('exy-dir', type=click.Path(file_okay=False))
def test(amount, agent, exy_dir):
    """
    Tests the neural network for the given amount of episodes.
    """
    exy = EXY(exy_dir, agent)
    exy.test(amount)


@rl.command()
@click.argument('exy-dir', type=click.Path(file_okay=False))
def plot(exy_dir):
    """
    Plots data from the stats files of each episode in the given EXY dir for
    analysis. Plots are saved to disk in each episode folder and the EXY dir
    itself.
    """
    exy = EXY(exy_dir, None)
    exy.plot_overall()


@rl.command()
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


@rl.command()
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
        subprocess.call(call, shell=True)

    shutil.rmtree(snp_dir)


@cli.group()
def ddonpai():
    log.info('Serial: %s', generate_now_serial_number())


@ddonpai.command()
@click.argument('ddonpai-dir')
@click.option('--show', is_flag=True)
def play(ddonpai_dir, show):
    ddon.play(ddonpai_dir, show=show)


if __name__ == '__main__':
    cli()
