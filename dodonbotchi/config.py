"""
This module defines DoDonBotchi's global configuration, including default
values for every config option. Functions to load, save, and create a default
configuration file are offered.

At the start of the program, `ensure_config` should be called to either create
a default configuration, if none exists, or load the existing one. The module
then stores an instance of the Config class in the field `CFG` which other
modules can easily import like `from dodonbotchi.config import CFG`.
"""

import logging as log
import json
import os

MAME_PATH = os.path.expanduser('~/mame/')
HOST = '127.0.0.1'
PORT = 32512

WINDOWED = True
NOVIDEO = False
NOAUDIO = True
NOTHROTTLE = False
SAVE_STATE = None
RENDER_SPRITES = "false"
RENDER_STATE = "false"
SHOW_INPUT = "true"
TICK_RATE = 2


class Config(dict):
    """
    Configuration class which mainly wraps around a dictionary to offer access
    to keys as members. Instead of `spam['eggs']`, it's possible to simply go
    `spam.eggs`.
    """

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        return super().__setitem__(key, val)

    def load_values(self, dic):
        """
        Loads every key-value pair from the given dictionary into the config.
        """
        for key, val in dic.items():
            self[key] = val

    def load(self, cfg_file):
        """
        Loads every key-value pair in the given json file into the config.
        """
        with open(cfg_file) as in_file:
            cfg_str = in_file.read()
            cfg_json = json.loads(cfg_str)
        self.load_values(cfg_json)

    def save(self, cfg_file):
        """
        Saves every key-value pair of this config into the given json file.
        """
        with open(cfg_file, 'w') as out_file:
            cfg_str = json.dumps(self, indent=4, sort_keys=True)
            out_file.write(cfg_str)


CFG = Config()


def get_default():
    """
    Creates and returns an instance of the `Config` class with the default
    value for each option.
    """
    dic = {
        'mame_path': MAME_PATH,
        'host': HOST,
        'port': PORT,
        'windowed': WINDOWED,
        'novideo': NOVIDEO,
        'noaudio': NOAUDIO,
        'nothrottle': NOTHROTTLE,
        'save_state': SAVE_STATE,
        'render_sprites': RENDER_SPRITES,
        'render_state': RENDER_STATE,
        'show_input': SHOW_INPUT,
        'tick_rate': TICK_RATE,
    }

    default = Config()
    default.load_values(dic)
    return default


def ensure_config(cfg_file):
    """
    Tests if the given cfg_file path points to a configuration file. If not, a
    default configuration will be written to that file. The file is then loaded
    into the `CFG` field.
    """
    if not os.path.exists(cfg_file):
        default = get_default()
        default.save(cfg_file)
        log.debug('Saved fresh default cfg to: %s', cfg_file)

    CFG.load(cfg_file)
