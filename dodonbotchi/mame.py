"""
This module implements classes related to giving access to MAME and DoDonPachi
as an OpenAI-Gym-like environment.
"""
import json
import logging as log
import math
import os
import random
import shutil
import socket
import subprocess

from time import sleep

import numpy as np

from jinja2 import Environment, FileSystemLoader
from PIL import Image

from dodonbotchi.config import CFG as cfg
from dodonbotchi.util import ensure_directories

SHELL = os.name == 'nt'

RECORDING_FILE = 'recording.inp'

PLUGIN_NAME = 'dodonbotchi_mame'

MAX_COMBO = 0x37
MAX_DISTANCE = 400  # Furthest distance two objects can have in 240x320


def get_action_str(vert=0, hori=0, shot=0, bomb=0):
    return '{}{}{}{}'.format(vert, hori, shot, bomb)


def get_plugin_path():
    """
    Gets the target directory to save the dodonbotchi plugin to relevant to the
    MAME home directory specified in the global config.
    """
    plugin_path = cfg.mame_path
    plugin_path = os.path.join(plugin_path, 'plugins')
    plugin_path = os.path.join(plugin_path, PLUGIN_NAME)
    return plugin_path


def write_plugin(**options):
    """
    Renders the templates for the code of the dodonbotchi plugin and writes it
    to the appropriate plugin directory.
    """
    plugin_path = get_plugin_path()
    ensure_directories(plugin_path)

    templates_path = os.path.join('dodonbotchi/plugin', PLUGIN_NAME)
    templates_env = Environment(loader=FileSystemLoader(templates_path))

    for template_name in os.listdir(templates_path):
        template_path = os.path.join(templates_path, template_name)
        if os.path.isfile(template_path):
            template = templates_env.get_template(template_name)
            rendered = template.render(plugin_name=PLUGIN_NAME, **options)

            template_target = os.path.join(plugin_path, template_name)
            with open(template_target, 'w') as out_file:
                out_file.write(rendered)


def generate_base_call():
    """
    Generates a list of parameters to start MAME with DoDonPachi which contain
    command line options corresponding to what's configured in the global
    configuration. The list gets returned and can be customised by appending
    further options.
    """
    call = ['mame', 'ddonpach', '-skip_gameinfo', '-pause_brightness', '1']

    if cfg.windowed:
        call.append('-window')

    if cfg.nothrottle:
        call.append('-nothrottle')

    if cfg.novideo:
        call.append('-video')
        call.append('none')

    if cfg.noaudio:
        call.append('-sound')
        call.append('none')

    if cfg.save_state:
        call.append('-state')
        call.append(cfg.save_state)

    return call


def render_avi(inp_file, avi_file, inp_dir=None, snp_dir=None):
    """
    Plays back input file at the given path recording it as a video to the
    given .avi file path. Optionally, recording and capture directories can be
    overridden using `inp_dir` and `snp_dir`.
    """
    write_plugin(mode='record', **cfg)

    call = generate_base_call()

    call.append('-plugin')
    call.append(PLUGIN_NAME)

    if inp_dir:
        call.append('-input_directory')
        call.append(inp_dir)

    call.append('-playback')
    call.append(inp_file)
    call.append('-exit_after_playback')

    if snp_dir:
        call.append('-snapshot_directory')
        call.append(snp_dir)

    call.append('-aviwrite')
    call.append(avi_file)

    return subprocess.call(call, shell=SHELL)


class Ddonpach:

    def __init__(self, recording=None, seed=None):
        self.inp_dir = None
        self.snp_dir = None
        self.sav_dir = None

        self.recording = recording

        self.process = None
        self.server = None
        self.client = None
        self.sfile = None
        self.waiting = True

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((cfg.host, cfg.port))
        self.server.listen()
        log.info('Started socket server on %s:%s', cfg.host, cfg.port)

        write_plugin(**cfg)

    def send_message(self, message, force=False):
        """
        Sends a message to the client, terminated by a newline.
        """
        if not force and not self.waiting:
            raise ValueError('Client is not waiting for new messages.')

        self.sfile.write('{}\n'.format(message))
        self.sfile.flush()
        self.waiting = False

    def send_command(self, command, force=False, **options):
        """
        Sends a command to the client in the form of a json object containing
        at least a `command` field and additional fields given in the
        **options.
        """
        message = {'command': command}
        for key, val in options.items():
            message[key] = val
        message = json.dumps(message)
        self.send_message(message, force=force)

    def send_action(self, action):
        """
        Sends an action to perform to the client. The given action must be a
        member of the DoDonPachiActions space.
        """
        self.send_command('action', inputs=action)

    def send_save_state(self, name):
        self.send_command('save', name=name)
        ack = self.read_message()
        assert ack['message'] == 'ACK'

    def send_load_state(self, name):
        self.send_command('load', name=name)
        ack = self.read_message()
        assert ack['message'] == 'ACK'

    def read_message(self):
        """
        Reads a message from the client, expecting it to be in one line and a
        json object. The method returns the message parsed as a dictionary.
        """
        if self.waiting:
            raise ValueError('Client is waiting for a message.')

        line = self.sfile.readline()
        self.waiting = True
        return json.loads(line)

    def read_gamestate(self):
        message = self.read_message()
        state_dic = message['state']
        return state_dic

    def get_snap(self):
        self.send_command('snap')
        ack = self.read_message()
        assert ack['message'] == 'ACK'

        snap_dir = os.path.join(self.snp_dir, 'ddonpach')
        snaps = list(sorted(os.listdir(snap_dir)))
        snaps = [snap for snap in snaps if snap != 'current.png']
        snap = snaps[-1]
        snap = os.path.join(snap_dir, snap)
        dest = os.path.join(snap_dir, 'current.png')
        shutil.copy(snap, dest)
        if os.path.exists(snap):
            os.remove(snap)
        ret = Image.open(dest)
        return ret

    def start_mame(self, avi=None):
        """
        Boots up MAME with the globally configured options and additionally
        setting it to record inputs this environment's respective folders for
        those.
        """
        ensure_directories(self.inp_dir, self.snp_dir)

        call = generate_base_call()
        call.append('-plugin')
        call.append(PLUGIN_NAME)

        abs_inp_dir = os.path.abspath(self.inp_dir)
        call.append('-input_directory')
        call.append(abs_inp_dir)
        if self.recording:
            call.append('-record')
            call.append(self.recording)

        abs_snp_dir = os.path.abspath(self.snp_dir)
        call.append('-snapshot_directory')
        call.append(abs_snp_dir)

        if self.sav_dir:
            abs_sav_dir = os.path.abspath(self.sav_dir)
            call.append('-state_directory')
            call.append(abs_sav_dir)

        if avi:
            call.append('-aviwrite')
            call.append(avi)

        self.process = subprocess.Popen(call, shell=SHELL)
        log.info('Started MAME with dodonbotchi ipc & dodonpachi.')
        log.info('Waiting for MAME to connect...')

        self.client, addr = self.server.accept()
        self.sfile = self.client.makefile(mode='rw')
        log.info('Accepted client from: %s', addr)

    def stop_mame(self):
        """
        Tries to gracefully terminate MAME by sending the client the kill
        command, but killing the process manually if the client does not
        terminate on its own.
        """
        if self.client and self.sfile:
            self.send_command('kill', force=True)

        sleep(1.5)

        for _ in range(10):
            if not self.process:
                break

            log.info('Waiting for MAME to die...')
            try:
                os.kill(self.process.pid, 0)
            except OSError:
                sleep(0.5)
                continue

            self.process = None

        if self.process:
            os.kill(self.process.pid, 9)
            self.process = None

        self.client = None
        self.sfile = None

    def __enter__(self):
        self.start_mame()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_mame()

    def close(self):
        """
        Kills MAME and closes the server socket.
        """
        if self.process:
            self.stop_mame()

        if self.server:
            self.server.close()

        self.process = None
        self.server = None
        self.client = None
        self.sfile = None
