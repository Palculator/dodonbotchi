"""
This module contains DoDonBotchi's brain and neuroevolutional code. It handles
learning, evolution, and controlling DoDonPachi in realtime.
"""
import json
import os
import random

from tensorflow.python import keras
from tensorflow.python.keras import models

from dodonbotchi import mame
from dodonbotchi.config import CFG as cfg

BRAIN_FILE = 'brain.keras'
PROPS_FILE = 'props.json'


def evaluate_brain(brain, mipc):
    pass


class Brain:
    @staticmethod
    def save(brain, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        brain_file = os.path.join(directory, BRAIN_FILE)
        brain.model.save(brain_file)

        props_dict = dict()
        props_dict['bid'] = brain.bid
        props_dict['score'] = brain.score

        props_file = os.path.join(directory, PROPS_FILE)
        with open(props_file) as out_file:
            json_content = json.dumps(props_dict, indent=4, sort_keys=True)
            out_file.write(json_content)

    @staticmethod
    def load(directory):
        if not os.path.exists(directory):
            print('Brain directory {} does not exist'.format(directory))
            return None

        brain_file = os.path.join(directory, BRAIN_FILE)
        if not os.path.exists(brain_file):
            print('Brain model file {} does not exist.'.format(directory))
            return None

        props_file = os.path.join(directory, PROPS_FILE)
        if not os.path.exists(props_file):
            print('Brain props file {} does not exist.'.format(directory))
            return None

        model = models.load_model(brain_file)
        with open(props_file, 'r') as in_file:
            props_content = in_file.read()
            props = json.loads(props_content)

        brain = Brain(props['bid'], model)
        return brain

    def __init__(self, bid, model):
        self.bid = bid
        self.model = model
        self.score = None

    def fit(self, states, inputs):
        epochs = cfg.epochs
        self.model.fit(states, inputs, epochs=epochs)

    def predict(self, observation):
        pass


class EXY:
    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(seed)
