import logging as log
import json
import os

MAME_PATH = os.path.expanduser('~/.mame/')
PLUGIN_NAME = 'dodonbotchi_api'
HOST = '127.0.0.1'
PORT = 32512


class Config(dict):
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __setitem__(self, key, val):
        return super().__setitem__(key, val)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __delitem__(self, key):
        return super().__delitem__(key)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def load(self, cfg_file):
        with open(cfg_file) as in_file:
            cfg_str = in_file.read()
            cfg_json = json.loads(cfg_str)

        for key, val in cfg_json.items():
            self[key] = val

    def save(self, cfg_file):
        with open(cfg_file, 'w') as out_file:
            cfg_str = json.dumps(self, indent=True, sort_keys=True)
            out_file.write(cfg_str)


cfg = Config()


def get_default():
    default = Config()

    default.mame_path = MAME_PATH
    default.plugin_name = PLUGIN_NAME
    default.host = HOST
    default.port = PORT

    return default


def ensure_config(cfg_file):
    if not os.path.exists(cfg_file):
        default = get_default()
        default.save(cfg_file)
        log.debug('Saved fresh default cfg to: %s', cfg_file)

    cfg.load(cfg_file)
